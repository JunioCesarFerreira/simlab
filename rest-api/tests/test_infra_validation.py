"""Input is refused at the edge, not deep inside.

Three behaviours were reproduced in review, each one a different way for bad
input to get past the API:

* ``bool("false")`` is ``True``, so a client sending the string *disabled* a
  host by enabling it;
* ``int("invalid")`` raises ``ValueError``, which no handler caught — the
  client got a 500 and the log got a traceback, for plainly bad input;
* ``access`` was allowed to be empty, so a host could be registered with
  nothing but a name and every later operation failed with "has no SSH
  address configured", far from its cause.

Unknown fields are the fourth: the routes said they were rejected and they
were silently dropped, so a client could send ``passwrod`` and get a 201
believing it had set something.
"""
import pytest

from tests.test_infra import RES_ID, sample_host, sample_resource

BASE = "/api/v1/infra"

REMOTE = {
    "name": "lab-99",
    "access": {
        "address": "10.0.0.9", "port": 2222, "username": "simlab",
        "auth_method": "password",
        "secret": {"alg": "crypto_box_seal", "key_id": "a" * 16,
                   "ciphertext": "AQ" * 60},
    },
}


def _create(client, **overrides):
    body = {**REMOTE, **overrides}
    return client.post(f"{BASE}/hosts", json=body)


class TestBooleans:
    def test_the_string_false_does_not_enable_a_host(self, client, mock_factory):
        mock_factory.host_repo.get_by_name.return_value = None

        resp = _create(client, enabled="false")

        assert resp.status_code == 201
        assert mock_factory.host_repo.insert.call_args.args[0]["enabled"] is False

    def test_a_value_that_is_not_a_boolean_is_refused(self, client):
        resp = _create(client, enabled="sim")

        assert resp.status_code == 422
        assert "true or false" in resp.json()["detail"]

    def test_a_real_boolean_still_works(self, client, mock_factory):
        mock_factory.host_repo.get_by_name.return_value = None
        assert _create(client, enabled=False).status_code == 201


class TestPorts:
    def test_a_textual_port_is_a_422_not_a_500(self, client):
        resp = _create(client, access={**REMOTE["access"], "port": "invalid"})

        assert resp.status_code == 422
        assert "must be a number" in resp.json()["detail"]

    def test_a_port_out_of_range_is_refused(self, client):
        resp = _create(client, access={**REMOTE["access"], "port": 99999})

        assert resp.status_code == 422
        assert "65535" in resp.json()["detail"]

    def test_a_container_port_is_checked_too(self, client, mock_factory):
        mock_factory.host_repo.get_by_name.return_value = None

        resp = _create(
            client, container_access={"username": "root", "port": 70000})

        assert resp.status_code == 422

    def test_an_explicit_zero_is_not_read_as_absent(self, client, mock_factory):
        """`or 22` quietly turned a nonsense port into the default."""
        mock_factory.host_repo.get_by_name.return_value = None

        resp = _create(
            client, container_access={"username": "root", "port": 0})

        assert resp.status_code == 422


class TestEnums:
    def test_an_unknown_auth_method_is_refused(self, client):
        resp = _create(client, access={**REMOTE["access"], "auth_method": "magic"})

        assert resp.status_code == 422
        assert "key" in resp.json()["detail"]

    def test_an_unknown_address_mode_is_refused(self, client):
        resp = _create(
            client, container_access={"username": "root", "address_mode": "magic"})

        assert resp.status_code == 422

    def test_an_unknown_discovery_mode_is_refused(self, client):
        resp = _create(client, discovery={"mode": "magic"})

        assert resp.status_code == 422


class TestUnknownFields:
    @pytest.mark.parametrize("block,field", [
        ("access", "passwrod"),
        ("container_access", "usernam"),
        ("discovery", "filtro"),
    ])
    def test_a_misspelled_field_is_named_in_the_refusal(
        self, client, block, field
    ):
        """Dropping it silently let a client believe it had set something."""
        body = {**REMOTE["access"]} if block == "access" else {}
        body[field] = "x"

        resp = _create(client, **{block: body})

        assert resp.status_code == 422
        assert field in resp.json()["detail"]


class TestReachability:
    def test_a_host_with_only_a_name_is_refused(self, client):
        resp = client.post(f"{BASE}/hosts", json={"name": "lab-99"})

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "access.address" in detail and "access.username" in detail

    def test_the_missing_field_is_named(self, client):
        resp = _create(
            client, access={k: v for k, v in REMOTE["access"].items()
                            if k != "address"})

        assert resp.status_code == 422
        assert "access.address" in resp.json()["detail"]

    def test_a_complete_registration_passes(self, client, mock_factory):
        mock_factory.host_repo.get_by_name.return_value = None
        assert _create(client).status_code == 201


# ── The gaps the third review reproduced ───────────────────────────────────

class TestEveryRouteGoesThroughTheSameRules:
    """R13 was corrected in some blocks and not others.

    Each case below is one the review reproduced against the running API:
    the rule existed, and the route did not use it.
    """

    def test_a_textual_port_on_a_worker_endpoint_is_a_422_not_a_500(
        self, client, mock_factory
    ):
        """``int("abc")`` raised ``ValueError`` from inside the mapper, which
        no handler caught: a 500 and a traceback for plainly bad input."""
        resp = client.post(
            f"{BASE}/hosts/507f1f77bcf86cd799430001/resources",
            json={"name": "cooja9",
                  "endpoint": {"address": "10.0.0.9", "port": "abc"}},
        )

        assert resp.status_code == 422
        assert "endpoint.port" in resp.json()["detail"]

    def test_the_string_false_does_not_enable_a_worker(
        self, client, mock_factory
    ):
        """The one route that still called ``bool()`` directly.

        ``bool("false")`` is True, so a client asking for a worker to leave
        the pool put it *into* the pool.
        """
        mock_factory.resource_repo.get.return_value = sample_resource()
        mock_factory.host_repo.get.return_value = sample_host()

        resp = client.put(
            f"{BASE}/resources/{RES_ID}", json={"enabled": "false"})

        if resp.status_code == 200:
            # Accepted, but read correctly.
            assert mock_factory.resource_repo.set_enabled.call_args.args[1] is False
        else:
            assert resp.status_code == 422

    def test_a_nonsense_enabled_value_is_refused(self, client, mock_factory):
        mock_factory.resource_repo.get.return_value = sample_resource()
        mock_factory.host_repo.get.return_value = sample_host()

        resp = client.put(f"{BASE}/resources/{RES_ID}", json={"enabled": "sim"})

        assert resp.status_code == 422

    def test_an_unknown_field_at_the_root_of_a_host_is_refused(self, client):
        """`passwrod` was dropped and answered with a 201, so the client
        believed it had set a credential."""
        resp = _create(client, passwrod="hunter2")

        assert resp.status_code == 422
        assert "passwrod" in resp.json()["detail"]

    def test_an_unknown_field_at_the_root_of_a_worker_is_refused(self, client):
        resp = client.post(
            f"{BASE}/hosts/507f1f77bcf86cd799430001/resources",
            json={"name": "cooja9", "endpoint": {"address": "10.0.0.9"},
                  "remot_dir": "/opt/cooja"},
        )

        assert resp.status_code == 422
        assert "remot_dir" in resp.json()["detail"]

    def test_an_unknown_field_inside_an_endpoint_is_refused(self, client):
        resp = client.post(
            f"{BASE}/hosts/507f1f77bcf86cd799430001/resources",
            json={"name": "cooja9",
                  "endpoint": {"address": "10.0.0.9", "prot": 22}},
        )

        assert resp.status_code == 422

    def test_a_fractional_port_is_refused_rather_than_truncated(self, client):
        """``int(22.9)`` is 22: the typo became a different port and the
        response reported 22 as though that had been asked for."""
        resp = _create(client, access={**REMOTE["access"], "port": 22.9})

        assert resp.status_code == 422
        assert "whole number" in resp.json()["detail"]

    def test_a_whole_float_is_still_accepted(self, client, mock_factory):
        """JSON has one number type; 2222.0 is how some clients send 2222."""
        mock_factory.host_repo.get_by_name.return_value = None

        resp = _create(client, access={**REMOTE["access"], "port": 2222.0})

        assert resp.status_code == 201

    def test_a_boolean_port_is_refused(self, client):
        """``int(True)`` is 1, a valid port."""
        resp = _create(client, access={**REMOTE["access"], "port": True})

        assert resp.status_code == 422

    def test_a_null_address_does_not_become_the_string_none(self, client):
        """``str(None)`` is ``"None"``, and that is what got stored — every
        later connection attempt then resolved that name."""
        resp = _create(client, access={**REMOTE["access"], "address": None})

        assert resp.status_code == 422
        assert "access.address" in resp.json()["detail"]

    def test_a_null_username_is_refused_too(self, client):
        resp = _create(client, access={**REMOTE["access"], "username": None})

        assert resp.status_code == 422


class TestABlockIsCheckedBeforeItsFields:
    """R13's remainder: a shape error must be a 422, not a traceback.

    Every rule enumerates a block's fields, and enumerating something that is
    not a mapping raises `TypeError` — which no route catches. The shape has
    to be checked first.
    """

    def test_a_scalar_where_an_endpoint_belongs(self, client):
        resp = client.post(
            f"{BASE}/hosts/507f1f77bcf86cd799430001/resources",
            json={"name": "cooja9", "endpoint": 123},
        )

        assert resp.status_code == 422
        assert "endpoint" in resp.json()["detail"]

    def test_a_scalar_where_access_belongs(self, client):
        resp = _create(client, access=42)

        assert resp.status_code == 422
        assert "access" in resp.json()["detail"]

    def test_a_scalar_where_container_access_belongs(self, client):
        resp = _create(client, container_access=42)

        assert resp.status_code == 422

    def test_a_scalar_where_discovery_belongs(self, client):
        resp = _create(client, discovery="auto")

        assert resp.status_code == 422

    def test_a_list_where_a_connection_belongs(self, client):
        resp = client.post(
            f"{BASE}/hosts/507f1f77bcf86cd799430001/resources",
            json={"name": "cooja9", "endpoint": {"address": "10.0.0.9"},
                  "connection": ["port", 22]},
        )

        assert resp.status_code == 422

    def test_a_nested_object_where_an_address_belongs(self, client):
        """It was accepted and stored as the text ``"{'a': 1}"``, and every
        later connection attempt resolved that name."""
        resp = client.post(
            f"{BASE}/hosts/507f1f77bcf86cd799430001/resources",
            json={"name": "cooja9", "endpoint": {"address": {"a": 1}}},
        )

        assert resp.status_code == 422
        assert "endpoint.address" in resp.json()["detail"]

    def test_a_null_block_means_nothing_to_say(self, client, mock_factory):
        """Distinct from a malformed one: the field-level rules already say
        what an absent override means."""
        mock_factory.host_repo.get_by_name.return_value = None

        resp = _create(client, discovery=None)

        assert resp.status_code == 201
