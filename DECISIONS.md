# Engineering Decisions

## Feature engineering: log1p instead of log10
The model was trained with `math.log1p(amount_sar)`, so the serving layer must reuse that transform exactly. A different transform changes the feature distribution and can silently drift scores even when the model artifact itself is unchanged. The service normalizes the feature names and casing at the boundary to keep the training and serving contracts stable.

## Protocol-based model abstraction
The `Model` protocol in `service/scorer.py` keeps the scoring service independent from any concrete ML library. This follows the dependency rule from the course: domain logic remains pure, while the adapter layer owns the joblib/sklearn integration. It allows the service to be tested with a fixed stub model without pulling in sklearn at all.

## Fail-closed scoring behaviour
When model inference fails or a malformed feature set reaches the scorer, the service returns `0.0` and logs a warning rather than propagating a stack trace to the caller. This is intentionally fail-safe: a scoring failure defaults to the lowest risk signal instead of silently turning the service into a backdoor for unsafe decisions.

## Decision thresholds
The default policy keeps a review band from 0.70 to 0.85 and blocks above 0.85. This creates a practical operating band for human review rather than treating fraud scoring as a binary classifier with a single threshold. It is aligned to the business needs described in the module: not every high score is a blockable event without context.

## API design choices
The API exposes the minimum required request/response contract with strict validation (`extra='forbid'`, bounded field lengths, positive finite amounts) to prevent malformed requests from reaching the model. Structured JSON logs and a trace ID allow incidents to be correlated in production without exposing internal stack traces.

## Container and deployment choices
The Docker image uses a multi-stage build to separate dependency installation from the runtime image, then runs as a non-root user. This keeps the container smaller and safer in shared environments. The health check probes `/v1/ready` and the compose file requires the service to be healthy before dependent services start.
