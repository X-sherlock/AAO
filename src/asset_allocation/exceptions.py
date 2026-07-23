class AllocationError(RuntimeError):
    """Base class for explicit, non-silent pipeline failures."""


class DataValidationError(AllocationError):
    """Raised when source data or temporal alignment is invalid."""


class InfeasibleProjectionError(AllocationError):
    """Raised when the configured hard-constraint set is infeasible."""


class OptionalDependencyError(AllocationError):
    """Raised when an optional training dependency is unavailable."""
