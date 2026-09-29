from app.security.config import (
    JWTConfig,
    JWTConfigError,
    get_jwt_config,
    reload_jwt_config,
)
from app.security.disclaimer import DISCLAIMER_PREFIX, DisclaimerService, disclaimer
from app.security.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiry,
    hash_password,
    verify_password,
)
from app.security.rbac import (
    ADMIN_LAWYER_ROLES,
    ADMIN_ROLES,
    CLIENT_ROLES,
    LAWYER_ROLES,
    RoleChecker,
    attach_user_to_request,
    get_current_user,
    get_optional_user,
    require_admin,
    require_lawyer,
    require_roles,
)
from app.security.sensitive_filter import detect_high_risk, mask_pii, sanitize_input

__all__ = [
    "JWTConfig",
    "JWTConfigError",
    "get_jwt_config",
    "reload_jwt_config",
    "DisclaimerService",
    "disclaimer",
    "DISCLAIMER_PREFIX",
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "get_token_expiry",
    "get_current_user",
    "get_optional_user",
    "require_roles",
    "require_admin",
    "require_lawyer",
    "RoleChecker",
    "attach_user_to_request",
    "ADMIN_ROLES",
    "LAWYER_ROLES",
    "CLIENT_ROLES",
    "ADMIN_LAWYER_ROLES",
    "mask_pii",
    "detect_high_risk",
    "sanitize_input",
]
