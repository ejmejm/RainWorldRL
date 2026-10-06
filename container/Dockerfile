# Runtime image for Rain World on Linux: Wine (64-bit), a virtual X display, Mesa (llvmpipe /
# lavapipe on the CPU, d3d12 on WSL2), DXVK, VirtualGL for GPU rendering, and SteamCMD for
# downloading the game. The game itself and the Wine prefix live outside the image (both need
# to be writable).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN dpkg --add-architecture i386 \
 && echo steam steam/question select "I AGREE" | debconf-set-selections \
 && echo steam steam/license note '' | debconf-set-selections \
 && sed -i 's/^Components: main restricted$/Components: main restricted multiverse/' /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      wine wine64 xvfb xauth libgl1 libgl1-mesa-dri mesa-utils mesa-vulkan-drivers libvulkan1 \
      steamcmd ca-certificates procps curl \
 && curl -fsSL -o /tmp/vgl.deb https://github.com/VirtualGL/virtualgl/releases/download/3.1.5/virtualgl_3.1.5_amd64.deb \
 && apt-get install -y --no-install-recommends /tmp/vgl.deb && rm /tmp/vgl.deb \
 && rm -rf /var/lib/apt/lists/*

# GPU rendering (renderer = "virtualgl"): VirtualGL redirects the game's GLX rendering to an EGL
# device. libglvnd finds NVIDIA's EGL library (bound in by apptainer --nv) through this vendor file.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
 && echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' \
      > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# DXVK (D3D11 -> Vulkan) for renderer = "cpu": on lavapipe it renders ~1.5x faster than Wine's own
# D3D11 -> OpenGL on llvmpipe. The launcher installs it into each new Wine prefix when /opt/dxvk exists.
# 2.7.1 is the newest that runs on Wine 9.0: DXVK 3.x needs VK_KHR_load_store_op_none, which Wine 9.0
# does not expose.
ARG DXVK_VERSION=2.7.1
ADD https://github.com/doitsujin/dxvk/releases/download/v${DXVK_VERSION}/dxvk-${DXVK_VERSION}.tar.gz /tmp/dxvk.tar.gz
RUN tar -C /opt -xzf /tmp/dxvk.tar.gz && mv /opt/dxvk-${DXVK_VERSION} /opt/dxvk && rm /tmp/dxvk.tar.gz

ENV PATH=/usr/games:/opt/VirtualGL/bin:$PATH
