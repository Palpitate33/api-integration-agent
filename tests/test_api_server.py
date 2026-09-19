"""Backend API 单元测试（TestClient，不启动真实服务器、不调用真实 DeepSeek）。

覆盖要求场景：
    1. GET /health                      8. Pipeline tests_failed
    2. 正常 POST /api/integrations/run  9. Pipeline exception
    3. 缺少必要参数                    10. PipelineResult JSON serialization
    4. 非法 project path               11. response 中没有 API Key
    5. path traversal                  12. Backend 不修改真实 repository
    6. 不存在的 project                13. CORS 行为
    7. Pipeline 正常返回                14. 错误响应结构
                                        15. demo_mode 安全边界（仅固定 Demo 组合）
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


def _diagnostic_text(payload) -> str:
    """递归取出所有 error / warnings 字段的**解码后**文本。

    只在诊断字段里查内部信息，因为那正是本次修复负责的位置：其余字段是业务数据
    （例如 python_files 里的相对路径 "pyproject.toml"、project.root 的绝对路径），
    对它们做全局字符串匹配会把正常数据误判成泄露。
    """
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"error", "warnings"}:
                found.extend(_flatten(value))
            else:
                found.append(_diagnostic_text(value))
    elif isinstance(payload, list):
        found.extend(_diagnostic_text(item) for item in payload)
    return "\n".join(found)


def _flatten(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for entry in value for item in _flatten(entry)]
    return []


def assert_http_response_is_sanitized(response) -> None:
    """HTTP 响应不得出现 traceback / 内部包路径，诊断字段里不得出现绝对路径。

    PipelineResult 是 response_model，会原样序列化给浏览器；这个断言是
    "客户端看不到服务端内部实现信息"的守门人。

    两个反直觉之处，都是实测踩出来的：
      1. 响应是 JSON，反斜杠会被转义成 `\\\\`。直接拿 str(EXAMPLES) 去匹配
         response.text 永远匹配不到——**泄露了也通过**（假阴性）。所以这里比对的是
         response.json() 里解码后的字段，而不是 response.text 的原文。
      2. 不能对整份响应做全局路径匹配：project.root 按契约就是绝对路径
         （见 scanner.py 的 ProjectStructure.root），它是正常业务数据。
    """
    assert "Traceback" not in response.text  # 无需转义，直接查原文
    assert "site-packages" not in response.text

    diagnostics = _diagnostic_text(response.json())
    assert "Traceback" not in diagnostics
    assert 'File "' not in diagnostics
    assert "site-packages" not in diagnostics
    for path in (EXAMPLES, Path(__file__).resolve().parent.parent):
        assert str(path) not in diagnostics


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


def test_stage_error_response_carries_no_internal_details(monkeypatch) -> None:
    """回归：阶段异常返回 200 + status=error，响应里不得有 traceback / 绝对路径。

    PipelineResult 是 response_model，_stage_error 把结果原样交给浏览器。
    修复前 error 是 "stage: ExceptionClass: message"、warnings 里直接塞
    traceback.format_exc()，两者都会把本地绝对路径与源码行送给客户端。

    必须让异常在 **stage 内部**抛出才走得到 _stage_error：替换整个 run_pipeline
    只会撞上 app 层的 500 兜底（那条路径由 test_pipeline_exception_becomes_500
    覆盖）。所以这里注入一个必然抛异常的 planner。
    """
    leaky_path = EXAMPLES / "demo_project" / "pyproject.toml"
    leaky_message = (
        f"cannot read {leaky_path}\n"
        "Traceback (most recent call last):\n"
        '  File "D:\\repo\\src\\integration_agent\\agent\\deepseek_planner.py", line 88\n'
        f"RuntimeError: boom at {leaky_path}"
    )

    class BoomPlanner:
        def plan(self, state):
            raise RuntimeError(leaky_message)

    monkeypatch.setattr(APP_MODULE, "DeepSeekPlanner", lambda *a, **k: BoomPlanner())
    monkeypatch.setattr(APP_MODULE, "DeepSeekLLMClient", lambda **kwargs: object())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")

    response = client.post("/api/integrations/run", json={**VALID_PAYLOAD, "use_llm_planner": True})

    assert response.status_code == 200  # 阶段失败是结构化 error，不是 HTTP 错误
    body = response.json()
    assert body["status"] == "error"
    assert body["failed_stage"] == "plan"
    assert body["error"] == "Integration failed during plan."
    assert body["api"]["name"] == "Demo Petstore API"  # 上游结果仍然保留
    # 异常原文（含绝对路径与伪造的 traceback 段落）整体不进入客户端可见字段
    assert "boom at" not in response.text
    assert "runtimeerror" not in response.text.lower()
    assert_http_response_is_sanitized(response)

    # 已知且**预先存在**的行为，不在本次 traceback 修复范围内：ProjectStructure.root
    # 按 scanner.py 的契约就是项目根的绝对路径，前端也没有消费它。这里显式钉住它，
    # 而不是在断言里绕开——否则"响应里有绝对路径"这件事就被悄悄藏起来了。
    assert body["project"]["root"] == str(DEMO_PROJECT_PATH)


# ------------------------------------------ 场景 15：demo_mode 安全边界


def test_demo_mode_rejected_for_non_demo_target() -> None:
    """demo_mode 只允许固定 Demo 组合，其余一律 400。"""
    payload = {"api_spec": "openapi/petstore.yaml", "project_path": "demo_project/.."}
    response = client.post("/api/integrations/run", json={**payload, "demo_mode": True})
    assert response.status_code == 400
    # 越界路径先被路径校验拦截（同一路径永远先过 paths.resolve_allowed）
    assert response.json()["error"]["code"] == "INVALID_PROJECT_PATH"


def test_demo_mode_rejected_when_target_not_the_fixed_demo() -> None:
    """路径合法（spec 是文件、project 是目录）但不是固定 Demo 组合时，demo_mode 必须被拒绝。"""
    response = client.post(
        "/api/integrations/run",
        json={"api_spec": "openapi/petstore.yaml", "project_path": "openapi", "demo_mode": True},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DEMO_MODE_NOT_ALLOWED"
    assert "demo_mode" in response.json()["error"]["message"]


def test_demo_mode_injects_deterministic_failure() -> None:
    """开启 demo_mode 后注入确定性失败：初始测试必然失败，不可能直接 passed。"""
    response = client.post(
        "/api/integrations/run",
        json={**VALID_PAYLOAD, "demo_mode": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "tests_failed"
    loop = body["repair_loop_result"]
    assert loop["repair_plans"], "初始测试应失败并产生 RepairPlan"
    assert loop["status"] != "passed"


def test_demo_mode_defaults_off(healthy_run) -> None:
    """不传 demo_mode 时行为与之前完全一致（普通请求不受影响）。"""
    assert healthy_run.status_code == 200
    assert healthy_run.json()["status"] == "passed"


def test_demo_mode_does_not_touch_repository() -> None:
    """demo_mode 只改内存产物，真实仓库文件不变。"""
    before = _snapshot_dir(DEMO_PROJECT_PATH)
    client.post("/api/integrations/run", json={**VALID_PAYLOAD, "demo_mode": True})
    after = _snapshot_dir(DEMO_PROJECT_PATH)
    assert before == after


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


# ------------------------------------- 场景 16：LLM Planner 接线（不调用真实 API）


@pytest.fixture
def capture_pipeline(monkeypatch) -> dict:
    """替换 run_pipeline，只记录被注入的组件，不执行真实 Pipeline。

    这样可以在**完全不接触 DeepSeek**的前提下验证接线：哪个开关替换了哪个组件。
    """
    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        captured["_args"] = args
        return pipeline.PipelineResult(
            status="passed", api=None, project=None, repair_loop_result=None
        )

    monkeypatch.setattr(APP_MODULE, "run_pipeline", fake_run)
    return captured


@pytest.fixture
def fake_deepseek(monkeypatch) -> dict:
    """把 app 模块里的 DeepSeekLLMClient 换成不需要 Key、不发网络的假客户端。"""
    created: dict = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

        def generate(self, prompt: str) -> str:
            return "{}"

    monkeypatch.setattr(APP_MODULE, "DeepSeekLLMClient", FakeClient)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")
    return created


def test_llm_switches_are_all_off_by_default(capture_pipeline) -> None:
    """默认三个开关都关：注入的仍是确定性组件。"""
    client.post("/api/integrations/run", json=VALID_PAYLOAD)

    assert "planner" not in capture_pipeline
    assert "repair_applier" not in capture_pipeline


def test_use_llm_alone_does_not_replace_planner(capture_pipeline, fake_deepseek) -> None:
    """兼容性：use_llm 保持既有语义——只影响修复，不改变 Planner。"""
    client.post("/api/integrations/run", json={**VALID_PAYLOAD, "use_llm": True})

    assert "repair_applier" in capture_pipeline
    assert "planner" not in capture_pipeline, "use_llm 不应改变既有规划行为"


def test_use_llm_repair_replaces_only_repair(capture_pipeline, fake_deepseek) -> None:
    client.post("/api/integrations/run", json={**VALID_PAYLOAD, "use_llm_repair": True})

    assert "repair_applier" in capture_pipeline
    assert "planner" not in capture_pipeline


def test_use_llm_planner_replaces_only_planner(capture_pipeline, fake_deepseek) -> None:
    """开启 Planner 开关时替换 Planner，且不连带启用修复。"""
    client.post("/api/integrations/run", json={**VALID_PAYLOAD, "use_llm_planner": True})

    assert "planner" in capture_pipeline
    assert "repair_applier" not in capture_pipeline
    from integration_agent.agent import DeepSeekPlanner

    assert isinstance(capture_pipeline["planner"], DeepSeekPlanner)


def test_planner_and_repair_can_be_enabled_together(capture_pipeline, fake_deepseek) -> None:
    """两个开关互相独立，可以同时开启，且共用一个无状态 LLM 客户端。"""
    client.post(
        "/api/integrations/run",
        json={**VALID_PAYLOAD, "use_llm_planner": True, "use_llm_repair": True},
    )

    assert "planner" in capture_pipeline
    assert "repair_applier" in capture_pipeline
    assert capture_pipeline["planner"].llm is capture_pipeline["repair_applier"].llm_client


def test_llm_planner_without_key_returns_structured_error(monkeypatch) -> None:
    """未配置 Key 时返回结构化 400，且不泄漏任何 Key 信息。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    response = client.post("/api/integrations/run", json={**VALID_PAYLOAD, "use_llm_planner": True})

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "LLM_NOT_CONFIGURED"
    assert "DEEPSEEK_API_KEY" in body["error"]["message"]
    assert "sk-" not in response.text


def test_planner_failure_is_not_silently_fallen_back(monkeypatch) -> None:
    """Planner 失败必须如实返回 status=error / failed_stage=plan，不得静默退回确定性规划。

    这里刻意使用**真实的** DeepSeekPlanner（只把底层 LLM 客户端换成会抛 401 的假实现），
    因此验证的是真实接线路径：DeepSeek 报 401 时，错误里夹带的密钥在到达客户端之前
    就已经被 Planner 脱敏。
    """

    class FailingLLM:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("DeepSeek API 返回 HTTP 401：invalid api key sk-live-shouldnotleak")

    monkeypatch.setattr(APP_MODULE, "DeepSeekLLMClient", lambda **kwargs: FailingLLM())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-not-a-real-key")

    response = client.post("/api/integrations/run", json={**VALID_PAYLOAD, "use_llm_planner": True})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["failed_stage"] == "plan"
    assert body["plan"] is None, "失败时不得退回确定性计划"
    # 异常原文整体不进入客户端可见字段：既没有密钥，也没有被脱敏后的残余。
    # 这与 Planner 层的脱敏是两道独立防线，不能互相替代。
    assert "sk-live-shouldnotleak" not in response.text
    assert "invalid api key" not in response.text
    assert body["error"] == "Integration failed during plan."
    assert_http_response_is_sanitized(response)
