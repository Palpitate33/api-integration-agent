"""API 数据模型单元测试。"""

from integration_agent.api.schema import APIAuth, APIEndpoint, APIInfo


def test_api_info_defaults() -> None:
    info = APIInfo(name="Demo", version="1.0")
    assert info.base_url is None
    assert info.description is None
    assert info.auth is None
    assert info.endpoints == []


def test_api_endpoint_defaults() -> None:
    endpoint = APIEndpoint(method="GET", path="/pets")
    assert endpoint.operation_id is None
    assert endpoint.summary is None
    assert endpoint.tags == []
    assert endpoint.parameters == []
    assert endpoint.responses == []
    assert endpoint.request_body is None


def test_api_auth_fields() -> None:
    auth = APIAuth(type="apiKey", name="X-API-Key", location="header")
    assert auth.scheme is None
    assert auth.bearer_format is None
    assert auth.flows is None


def test_model_dump_is_serializable() -> None:
    info = APIInfo(name="Demo", version="1.0")
    info.endpoints.append(APIEndpoint(method="GET", path="/pets"))
    data = info.model_dump()
    assert data["name"] == "Demo"
    assert data["endpoints"][0]["method"] == "GET"
