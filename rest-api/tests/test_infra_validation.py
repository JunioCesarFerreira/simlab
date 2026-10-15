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
