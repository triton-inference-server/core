#pragma once

enum class FailureReason {
    REJECTED,
    CANCELED,
    BACKEND,
    TIMEOUT,
    OTHER
};

