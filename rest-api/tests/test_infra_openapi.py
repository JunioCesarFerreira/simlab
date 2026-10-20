"""The published contract has to be the one the code enforces.

These routes take a ``dict`` body on purpose: the rules they apply — absent
versus null versus a value, a credential that must arrive sealed — cannot be
expressed by a model without losing them. But `Body(..., description="See
HostCreateDto")` published an arbitrary object, so the OpenAPI document and
the Swagger page said nothing true, and a generated client had nothing to go
on. Three reviews asked for this.

The schema is therefore built from the same tuples the mapper enforces, and
this file is what makes that trustworthy: a field added to one and not the
other fails here. Without it, a hand-written schema is a second thing to keep
in sync, which is worse than none.
"""
import pytest

from api.mappers import infra as mapper
from api.schemas import infra as schemas

BASE = "/api/v1/infra"


def _properties(schema: dict) -> set[str]:
    return set(schema.get("properties", {}))


class TestTheSchemaMatchesTheMapper:
    @pytest.mark.parametrize("schema,fields", [
        (schemas.HOST_CREATE, mapper._HOST_FIELDS),
        (schemas.HOST_UPDATE, mapper._HOST_FIELDS),
        (schemas.RESOURCE_CREATE, mapper._RESOURCE_FIELDS),
        (schemas.RESOURCE_UPDATE, mapper._RESOURCE_FIELDS),
    ])
    def test_the_root_accepts_exactly_what_the_mapper_knows(self, schema, fields):
        assert _properties(schema) == set(fields)

    @pytest.mark.parametrize("block,fields", [
        ("access", mapper._ACCESS_FIELDS),
        ("container_access", mapper._CONTAINER_ACCESS_FIELDS),
        ("discovery", mapper._DISCOVERY_FIELDS),
    ])
    def test_each_host_block_does_too(self, block, fields):
        assert _properties(
            schemas.HOST_CREATE["properties"][block]) == set(fields)

    @pytest.mark.parametrize("block,fields", [
        ("endpoint", mapper._ENDPOINT_FIELDS),
        ("connection", mapper._CONNECTION_FIELDS),
    ])
    def test_and_each_worker_block(self, block, fields):
        assert _properties(
            schemas.RESOURCE_CREATE["properties"][block]) == set(fields)

    def test_unknown_fields_are_refused_in_the_document_too(self):
        """``additionalProperties: false`` is the published form of
        ``_refuse_unknown``. Without it the document would promise to accept
        what the route answers 422 for."""
        for schema in (schemas.HOST_CREATE, schemas.HOST_UPDATE,
                       schemas.RESOURCE_CREATE, schemas.RESOURCE_UPDATE,
                       schemas.RESOURCE_ENABLED):
            assert schema["additionalProperties"] is False
        for block in ("access", "container_access", "discovery"):
            assert schemas.HOST_CREATE["properties"][block][
                "additionalProperties"] is False

    def test_the_port_range_is_the_mapper_s(self):
        port = schemas.HOST_CREATE["properties"]["access"]["properties"]["port"]
        assert port["minimum"] == mapper._MIN_PORT
        assert port["maximum"] == mapper._MAX_PORT

    def test_creation_states_what_it_requires(self):
        assert set(schemas.HOST_CREATE["required"]) == {"name", "access"}
        assert set(schemas.RESOURCE_CREATE["required"]) == {"name", "endpoint"}

    def test_an_update_requires_nothing(self):
        """Every field is optional, because absent means "leave it"."""
        assert "required" not in schemas.HOST_UPDATE
        assert "required" not in schemas.RESOURCE_UPDATE

    def test_a_credential_is_documented_as_sealed(self):
        secret = schemas.HOST_CREATE["properties"]["access"]["properties"][
            "secret"]
        assert secret["properties"]["alg"]["const"] == "crypto_box_seal"
        assert "null" in secret["type"], "null removes the stored credential"
        assert "Plaintext is refused" in secret["description"]


class TestTheDocumentCarriesIt:
    """And the routes actually publish it."""

    WRITES = [
        ("post", "/hosts"),
        ("put", "/hosts/{host_id}"),
        ("post", "/hosts/{host_id}/resources"),
        ("put", "/resources/{resource_id}"),
        ("put", "/resources/{resource_id}/connection"),
    ]

    def _schema_of(self, document, method, path):
        operation = document["paths"][f"/api/v1/infra{path}"][method]
        body = operation["requestBody"]["content"]["application/json"]["schema"]
        return body

    @pytest.mark.parametrize("method,path", WRITES)
    def test_the_body_is_described(self, client, method, path):
        document = client.get("/openapi.json").json()

        schema = self._schema_of(document, method, path)

        assert schema.get("additionalProperties") is False, (
            "the route still publishes an arbitrary object")
        assert schema.get("properties"), schema

    def test_a_reader_can_tell_what_a_null_means(self, client):
        """The three-way intent is the part a model cannot express, so the
        document has to say it in words."""
        document = client.get("/openapi.json").json()

        schema = self._schema_of(
            document, "put", "/resources/{resource_id}/connection")

        assert "inherits again" in schema["properties"]["connection"][
            "description"]

    def test_every_route_a_description_points_at_exists(self, client):
        """A description that names a route is part of the contract.

        ``SEALED_SECRET`` told readers to fetch the sealing key from
        ``GET /infra/public-key``, which is a 404 — the route is
        ``/infra/crypto/public-key``. Nothing caught it, because a prose
        mistake in a schema is still published as fact, and the first person
        to follow it is the one writing a client.

        So every ``GET|POST|PUT|PATCH|DELETE /path`` written in any
        description of the document has to resolve to a path the application
        serves.
        """
        import re

        document = client.get("/openapi.json").json()
        served = set(document["paths"])
        prefixed = {p.replace("/api/v1", "", 1) for p in served}

        # A path may end a sentence, so trailing punctuation is not part of
        # it; a dot *inside* one (a file extension) is.
        pattern = re.compile(
            r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+(/[A-Za-z0-9_/{}.-]*[A-Za-z0-9_/}-])")
        referenced: set[str] = set()

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "description" and isinstance(value, str):
                        referenced.update(pattern.findall(value))
                    else:
                        walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(document)
        assert referenced, "the guard would pass vacuously"
        unknown = sorted(
            path for path in referenced
            if path not in served and path not in prefixed)
        assert unknown == [], (
            f"described but not served: {unknown}")
