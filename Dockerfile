# Dev tooling image — ALL build/test/bench commands run inside this container.
# Usage: docker run --rm -v "$PWD":/work -w /work dataframe-dev bash -lc '<cmd>'
FROM node:22-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git binaryen \
 && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain stable --target wasm32-unknown-unknown \
 && rustup component add clippy rustfmt \
 && chmod -R a+rw "$RUSTUP_HOME" "$CARGO_HOME"

WORKDIR /work
