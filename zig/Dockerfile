# The engine image: TensorFold's native (Zig) engine with the DeepSeek-V4.1-Flash family of BertholomusAI's fork
# (TF_REPO at TF_REF), plus zig/patches/*.patch (git apply in the checkout's root, filename order): DeepSeek's own FP8 /
# MXFP4 checkpoint and the four-Spark ring's node order. Built on NVIDIA's PyTorch container, which every Spark already
# holds: its CUDA toolchain compiles the sm_121 fatbins, its NCCL 2.30.7, cuBLAS and libibverbs are what the engine
# loads at run time, and its Python with Pillow 12.3.0 decodes the picture formats other than PNG (the engine starts a
# python3 child for them). PyAV (ARG PYAV_VERSION) is added for video input: the server's python3 child decodes a
# video_url part into the frames the model reads. Structured output's libtfgrammar is compiled by zig from an xgrammar
# checkout (ARG XGRAMMAR_REF). No GPU is needed to build.
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
# CPU-only lost-wakeup and park/wake tests; no GPU or model weights needed.
RUN timeout 120s /opt/zig/zig test /tensorfold/zig/src/families/dsv41/idle_gate.zig -O ReleaseSafe -lc -j1
# CPU-only rendezvous tests over loopback (0035): workers that are ready first wait in their ports' backlogs.
RUN timeout 120s /opt/zig/zig test /tensorfold/zig/src/families/dsv41/link.zig -O ReleaseSafe -lc -j1
# Structured output (0036): libtfgrammar, TensorFold's C ABI over xgrammar (Apache-2.0), built by `zig build grammar`
# from xgrammar's v0.2.8 release (the one tf_grammar.cc is written against) with its submodules; the server loads it at
# the first structured request (TENSORFOLD_GRAMMAR_LIB). The checkout is deleted after the build; its licenses stay.
ARG XGRAMMAR_TAG=v0.2.8
ARG XGRAMMAR_REF=97787376faee5ed8466cfad57c99855e4ce2f6aa
RUN git clone -q --depth 1 --branch "${XGRAMMAR_TAG}" --recurse-submodules --shallow-submodules \
      https://github.com/mlc-ai/xgrammar /opt/xgrammar && \
    test "$(git -C /opt/xgrammar rev-parse HEAD)" = "${XGRAMMAR_REF}"
ARG SM=121
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
      -Dxgrammar=/opt/xgrammar --prefix /opt/tensorfold -j"$(nproc)" install native grammar && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && test -x /opt/tensorfold/bin/tf-dsv41-lanes && \
    test -x /opt/tensorfold/bin/tf-dsv41-rank-cache && test -f /opt/tensorfold/lib/libtfgrammar.so && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES && \
    cp /opt/xgrammar/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-Apache-2.0.txt && \
    cp /opt/xgrammar/NOTICE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-NOTICE.txt && \
    cp /opt/xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/dlpack-Apache-2.0.txt && \
    rm -rf /opt/xgrammar && \
    mkdir -p /opt/tensorfold/tools && cp -a tools/dsv41-zig/inputs tools/dsv41-zig/native /opt/tensorfold/tools/ && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold && \
    for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    rm -rf /tensorfold/.zig-cache /root/.cache/zig
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.engine=zig tf.ref=${TF_REF}
ENV PATH=/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH}
ENV TENSORFOLD_GRAMMAR_LIB=/opt/tensorfold/lib/libtfgrammar.so
WORKDIR /workspace
