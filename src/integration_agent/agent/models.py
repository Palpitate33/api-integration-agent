"""IntegrationPlan 及其子结构：Planner 的结构化输出契约。

设计约束：
    - 全部核心字段强类型化，Code Generator 可直接消费，无需再解析自由文本。
    - API 侧信息（TargetAPI）与仓库侧信息（RepositoryContext）严格分离。
    - summary / purpose / strategy / reason 等文本字段只承载"理由"，
      真正的参数（方法、路径、状态码、文件路径、依赖名）一律走独立的结构化字段。
"""

from typing import Literal

from pydantic import BaseModel, Field

from integration_agent.api import APIEndpoint


class TargetAPI(BaseModel):
    """API 侧信息（来源：api.schema.APIInfo）。"""

    name: str
    version: str
    base_url: str | None = None
    total_endpoints: int = 0  # API 声明的端点总数（可能多于本次集成的数量）


class RepositoryContext(BaseModel):
    """仓库侧信息（来源：repository.scanner.ProjectStructure + 受控的代码检索）。"""

    root: str
    name: str
    source_dirs: list[str] = Field(default_factory=list)
    test_dirs: list[str] = Field(default_factory=list)
    http_client: str | None = None  # 仓库已在使用的 HTTP 客户端库名
    http_client_module: str | None = None  # 该客户端被使用的位置（相对路径）
    existing_modules: list[str] = Field(default_factory=list)  # 与 API 领域相关的既有模块
    inspected_files: list[str] = Field(default_factory=list)  # 本次规划实际检索过的文件


class IntegrationStrategy(BaseModel):
    """集成策略：Code Generator 据此决定代码放置位置与实现方式。"""

    approach: Literal["new_client_module", "extend_existing_client"]
    summary: str
    client_module: str  # 客户端代码的目标路径
    http_client: str  # 将要使用的 HTTP 客户端库


class PlannedFile(BaseModel):
    """待新建的文件。"""

    path: str
    purpose: str
    kind: Literal["client", "models", "exceptions", "config", "package_init", "other"] = "other"


class FileModification(BaseModel):
    """待修改的既有文件。"""

    path: str
    reason: str
    changes: list[str] = Field(default_factory=list)


class DependencyRequirement(BaseModel):
    """需要增加的 Python 依赖。"""

    name: str
    version: str | None = None  # 版本约束，如 ">=0.27"
    reason: str = ""
    already_installed: bool = False  # 已在项目声明中，无需新增


class PlannedEndpoint(BaseModel):
    """本次集成实际要使用的端点（复用 api.schema.APIEndpoint）。"""

    endpoint: APIEndpoint
    purpose: str

    @property
    def method(self) -> str:
        return self.endpoint.method

    @property
    def path(self) -> str:
        return self.endpoint.path

    @property
    def operation_id(self) -> str | None:
        return self.endpoint.operation_id


class AuthenticationPlan(BaseModel):
    """认证方式与配置位置。"""

    scheme: str  # apiKey / http / oauth2 / openIdConnect
    strategy: str  # 凭据如何注入请求
    config_location: str  # 凭据从何处读取
    required_env_vars: list[str] = Field(default_factory=list)
    header_name: str | None = None  # apiKey / bearer 时的目标 header


class ErrorHandlingRule(BaseModel):
    """单个错误类别的处理策略。"""

    status_code: str  # "400" / "401" / ... / "5xx"
    category: Literal[
        "validation",
        "authentication",
        "authorization",
        "not_found",
        "rate_limit",
        "server_error",
    ]
    strategy: str
    retryable: bool = False
    documented_in_spec: bool = False  # OpenAPI 文档中是否声明了该响应


class ErrorHandlingPlan(BaseModel):
    """错误处理策略集合。"""

    exception_type: str  # 建议的自定义异常基类名
    rules: list[ErrorHandlingRule] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TestSpec(BaseModel):
    """一项待编写的测试。"""

    path: str
    purpose: str
    covers: list[str] = Field(default_factory=list)


class TestingStrategy(BaseModel):
    """测试策略：unit / integration / contract 三类。"""

    unit_tests: list[TestSpec] = Field(default_factory=list)
    integration_tests: list[TestSpec] = Field(default_factory=list)
    contract_tests: list[TestSpec] = Field(default_factory=list)


class Risk(BaseModel):
    """可能导致集成失败的问题。"""

    description: str
    severity: Literal["low", "medium", "high"]
    mitigation: str | None = None


class IntegrationPlan(BaseModel):
    """Planner 的完整输出：可被 Code Generator 直接消费的集成方案。"""

    integration_goal: str
    target_api: TargetAPI
    repository: RepositoryContext
    integration_strategy: IntegrationStrategy
    files_to_create: list[PlannedFile] = Field(default_factory=list)
    files_to_modify: list[FileModification] = Field(default_factory=list)
    dependencies: list[DependencyRequirement] = Field(default_factory=list)
    endpoints: list[PlannedEndpoint] = Field(default_factory=list)
    authentication: AuthenticationPlan | None = None
    error_handling: ErrorHandlingPlan
    testing_strategy: TestingStrategy
    assumptions: list[str] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
