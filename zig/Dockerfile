# The engine image: TensorFold's native (Zig) engine with the DeepSeek-V4.1-Flash family of BertholomusAI's fork
# (TF_REPO at TF_REF), plus zig/patches/*.patch (git apply in the checkout's root, filename order): DeepSeek's own FP8 /
# MXFP4 checkpoint and the four-Spark ring's node order. Built on NVIDIA's PyTorch container, which every Spark already
# holds: its CUDA toolchain compiles the sm_121 fatbins, its NCCL 2.30.7, cuBLAS and libibverbs are what the engine
# loads at run time, and its Python with Pillow 12.3.0 decodes the picture formats other than PNG (the engine starts a
# python3 child for them). PyAV (ARG PYAV_VERSION) is added for video input: the server's python3 child decodes a
# video_url part into the frames the model reads. No GPU is needed to build.
# Built by scripts/prepare.sh, never by hand: it passes the build arguments and a build context of this file and
# zig/patches only.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG PYAV_VERSION=18.1.0
RUN pip install --no-cache-dir "av==${PYAV_VERSION}" && python3 -c "import av, PIL; print('PyAV', av.__version__)"
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY zig/patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
ARG SM=121
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
      --prefix /opt/tensorfold -j"$(nproc)" install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && test -x /opt/tensorfold/bin/tf-dsv41-lanes && \
    test -x /opt/tensorfold/bin/tf-dsv41-rank-cache && \
    mkdir -p /opt/tensorfold/tools && cp -a tools/dsv41-zig/inputs tools/dsv41-zig/native /opt/tensorfold/tools/ && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold && \
    for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    rm -rf /tensorfold/.zig-cache /root/.cache/zig
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.engine=zig tf.ref=${TF_REF}
ENV PATH=/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH}
WORKDIR /workspace
