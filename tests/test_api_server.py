"""Backend API 单元测试（TestClient，不启动真实服务器、不调用真实 DeepSeek）。

覆盖要求场景：
    1. GET /health                      8. Pipeline tests_failed
    2. 正常 POST /api/integrations/run  9. Pipeline exception
    3. 缺少必要参数                    10. PipelineResult JSON serialization
    4. 非法 project path               11. response 中没有 API Key
    5. path traversal                  12. Backend 不修改真实 repository
    6. 不存在的 project                13. CORS 行为
    7. Pipeline 正常返回                14. 错误响应结构
"""

from importlib import import_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from integration_agent import pipeline
from integration_agent.api_server.app import app

# 注意：api_server/__init__.py 把包属性 app 绑定为 FastAPI 实例，
# 因此用 import_module 直接取 app 模块对象，供 monkeypatch 使用。
APP_MODULE = import_module("integration_agent.api_server.app")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
DEMO_PROJECT_PATH = EXAMPLES / "demo_project"

client = TestClient(app)

VALID_PAYLOAD = {
    "api_spec": "openapi/petstore.yaml",
    "project_path": "demo_project",
}


def _snapshot_dir(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def healthy_run():
    """真实 Pipeline 调用一次，供多个测试共享。"""
    return client.post("/api/integrations/run", json=VALID_PAYLOAD)


# ------------------------------------------------- 场景 1：health


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ------------------------------------ 场景 2/7/10：正常运行与序列化


def test_run_integration_returns_pipeline_result(healthy_run) -> None:
    response = healthy_run
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "passed"
    assert body["api"]["name"] == "Demo Petstore API"
    assert body["project"]["name"] == "demo-project"
    assert body["plan"] is not None
    assert body["artifacts"] is not None
    assert body["repair_loop_result"]["status"] == "passed"
    assert body["patch"] is not None
    assert body["patch"]["summary"]["created"] == 7
    assert body["patch"]["summary"]["modified"] == 2
    # 结构化 JSON 可被 PipelineResult 反序列化（round-trip）
    assert pipeline.PipelineResult.model_validate_json(response.text) is not None


# ------------------------------------------------ 场景 3：缺少参数


def test_missing_required_params() -> None:
    response = client.post("/api/integrations/run", json={"project_path": "demo_project"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"


# --------------------------------------- 场景 4-5：非法 / 越界路径


@pytest.mark.parametrize(
    "payload",
    [
        {"api_spec": "openapi/petstore.yaml", "project_path": "C:\\Users\\x"},
        {"api_spec": "openapi/petstore.yaml", "project_path": "/etc"},
        {"api_spec": "openapi/petstore.yaml", "project_path": "../../../secret"},
        {"api_spec": "openapi/petstore.yaml", "project_path": "demo_project/../.."},
    ],
)
def test_invalid_project_paths_rejected(payload: dict) -> None:
    response = client.post("/api/integrations/run", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_PROJECT_PATH"


def test_invalid_spec_path_rejected() -> None:
    response = client.post(
        "/api/integrations/run",
        json={"api_spec": "../../etc/passwd", "project_path": "demo_project"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_SPEC_PATH"


# ------------------------------------------- 场景 6：不存在的资源


def test_project_not_found() -> None:
    response = client.post(
        "/api/integrations/run",
        json={"api_spec": "openapi/petstore.yaml", "project_path": "no_such_project"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PROJECT_NOT_FOUND"


def test_spec_not_found() -> None:
    response = client.post(
        "/api/integrations/run",
        json={"api_spec": "openapi/no_such.yaml", "project_path": "demo_project"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SPEC_NOT_FOUND"


# ------------------------------------- 场景 8-9：Pipeline 状态与异常


def test_pipeline_tests_failed(monkeypatch) -> None:
    failed = pipeline.PipelineResult(
        status="tests_failed",
        api=None,
        project=None,
        repair_loop_result=None,
    )

    def fake_run(*args, **kwargs):
        return failed

    monkeypatch.setattr(APP_MODULE, "run_pipeline", fake_run)
    response = client.post("/api/integrations/run", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["status"] == "tests_failed"


def test_pipeline_exception_becomes_500(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail sk-abcdef123456")

    monkeypatch.setattr(APP_MODULE, "run_pipeline", boom)
    response = client.post("/api/integrations/run", json=VALID_PAYLOAD)
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "PIPELINE_ERROR"
    # 不向客户端返回 traceback / 异常原文
    assert "sk-abcdef" not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text


# ------------------------------------------- 场景 11：无 API Key 泄露


def test_no_api_key_in_response(monkeypatch, healthy_run) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-super-secret-xyz")
    response = client.post("/api/integrations/run", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "sk-super-secret-xyz" not in response.text
    assert "sk-" not in response.text


def test_llm_mode_without_key_returns_clean_error(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    payload = {**VALID_PAYLOAD, "use_llm": True}
    response = client.post("/api/integrations/run", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "LLM_NOT_CONFIGURED"
    assert "DEEPSEEK_API_KEY" in body["error"]["message"]
    assert "sk-" not in response.text


# -------------------------------------- 场景 12：不修改真实 repository


def test_repository_unchanged() -> None:
    before = _snapshot_dir(DEMO_PROJECT_PATH)
    client.post("/api/integrations/run", json=VALID_PAYLOAD)
    after = _snapshot_dir(DEMO_PROJECT_PATH)
    assert before == after


# --------------------------------------------------- 场景 13：CORS


def test_cors_allows_local_frontend_origin() -> None:
    response = client.options(
        "/api/integrations/run",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin() -> None:
    response = client.options(
        "/api/integrations/run",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


# ---------------------------------------------- 场景 14：错误响应结构


def test_error_response_structure() -> None:
    response = client.post(
        "/api/integrations/run",
        json={"api_spec": "openapi/petstore.yaml", "project_path": "demo_project/../../x"},
    )
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert isinstance(body["error"]["code"], str)
    assert isinstance(body["error"]["message"], str)


# --------------------------------------------------- 请求模型校验


def test_request_model_bounds() -> None:
    payload = {**VALID_PAYLOAD, "max_iterations": 99}
    response = client.post("/api/integrations/run", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
