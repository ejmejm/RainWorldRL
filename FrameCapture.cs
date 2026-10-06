using System;
using UnityEngine;

/// <summary>
/// Captures game frames at a configurable resolution for RL observation.
/// Reads what the main (Futile) camera rendered this frame from its target texture, scales it on the
/// GPU and reads back only the small frame (one GPU sync per capture; this matters with software
/// rendering). Rain World's camera always renders into a texture that a UI image draws to the screen
/// later in the frame, so the back buffer at capture time still holds the previous rendered frame.
///
/// While the agent is in control (<see cref="SetAgentInControl"/>) the camera renders straight into a
/// texture <see cref="RenderScale"/> x the frame size with the game's field of view, so far fewer
/// pixels are shaded than at 1366x768, and with <see cref="SMALL_WINDOW_ENV_VAR"/> set the window
/// shrinks to the frame size so compositing and presenting it is cheap too.
/// Uses lazy initialization to avoid creating textures before Unity is ready.
/// </summary>
public class FrameCapture : IDisposable
{
    private RenderTexture renderTexture;
    private Texture2D captureTexture;
    private RenderTexture screenCapture;  // Full-size GPU copy of the back buffer (camera without a target texture)
    private RenderTexture smallTarget;    // The camera's render target while the agent is in control
    private int savedWindowWidth;         // Window size before shrinking it; 0 while not shrunk
    private int savedWindowHeight;
    private FullScreenMode savedWindowMode;
    private int width;
    private int height;
    private int lastScreenWidth;
    private int lastScreenHeight;
    private bool disposed = false;
    private bool initialized = false;

    public int Width => width;
    public int Height => height;
    public int FrameSize => width * height * 3;

    /// <summary>If set (the Linux launcher sets it), the window shrinks to the frame size while the agent
    /// is in control. Nothing reads the window then, and under software rendering compositing and presenting
    /// a full-size window costs several ms per step.</summary>
    public const string SMALL_WINDOW_ENV_VAR = "RAINWORLD_RL_SMALL_WINDOW";

    /// <summary>The camera renders into a texture this many times the frame size while the agent is in
    /// control; the capture averages it down. 0 keeps the game's own full-size target.</summary>
    public int RenderScale { get; set; } = 2;

    private readonly bool shrinkWindow = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable(SMALL_WINDOW_ENV_VAR));

    /// <summary>
    /// Initializes the frame capture system with the specified dimensions.
    /// Textures are created lazily on first capture to avoid early initialization issues.
    /// </summary>
    public FrameCapture(int width, int height)
    {
        this.width = width;
        this.height = height;
        // Don't create textures here - Unity might not be ready
    }

    /// <summary>
    /// Ensures textures are created. Called lazily before first use.
    /// </summary>
    private void EnsureInitialized()
    {
        if (initialized)
            return;

        try
        {
            // Create render texture for capturing camera output
            renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            renderTexture.Create();

            // Create texture for reading pixels
            captureTexture = new Texture2D(width, height, TextureFormat.RGB24, false);

            initialized = true;
        }
        catch (Exception ex)
        {
            Debug.LogError($"[FrameCapture] Failed to initialize textures: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Resizes the capture textures to new dimensions.
    /// </summary>
    public void Resize(int newWidth, int newHeight)
    {
        if (newWidth == width && newHeight == height)
            return;

        width = newWidth;
        height = newHeight;

        // If already initialized, recreate textures
        if (initialized)
        {
            if (renderTexture != null)
            {
                renderTexture.Release();
                UnityEngine.Object.Destroy(renderTexture);
            }

            if (captureTexture != null)
            {
                UnityEngine.Object.Destroy(captureTexture);
            }

            renderTexture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            renderTexture.Create();
            captureTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
        }
        // The small target is resized by the next SetAgentInControl, not here: Resize runs inside the
        // camera's post-render, where the capture still has to read what was just rendered into it.
    }

    /// <summary>
    /// Called every frame. While the agent is in control the Futile camera renders into the small target
    /// (and the window shrinks, see <see cref="SMALL_WINDOW_ENV_VAR"/>); otherwise the game's own target and
    /// window size are restored. Cheap when nothing changes.
    /// </summary>
    public void SetAgentInControl(bool agent)
    {
        Camera cam = Futile.instance != null ? Futile.instance.camera : null;
        if (cam == null)
            return;

        int targetWidth = width * RenderScale;
        int targetHeight = height * RenderScale;
        if (smallTarget != null && (!agent || smallTarget.width != targetWidth || smallTarget.height != targetHeight))
            ReleaseSmallTarget();  // restores the game's own target

        if (agent && RenderScale > 0)
        {
            if (smallTarget == null)
            {
                smallTarget = new RenderTexture(targetWidth, targetHeight, 24, RenderTextureFormat.ARGB32);
                smallTarget.filterMode = FilterMode.Bilinear;
                smallTarget.Create();
            }
            if (cam.targetTexture != smallTarget)
            {
                cam.targetTexture = smallTarget;
                cam.aspect = (float)Futile.screen.pixelWidth / Futile.screen.pixelHeight;  // 1366x768 is not exactly 16:9
            }
        }

        if (agent && shrinkWindow && savedWindowWidth == 0)
        {
            savedWindowWidth = Screen.width;
            savedWindowHeight = Screen.height;
            savedWindowMode = Screen.fullScreenMode;
            Screen.SetResolution(width, height, FullScreenMode.Windowed);
        }
        else if (!agent && savedWindowWidth != 0)
        {
            Screen.SetResolution(savedWindowWidth, savedWindowHeight, savedWindowMode);
            savedWindowWidth = 0;
        }
    }

    private void ReleaseSmallTarget()
    {
        if (smallTarget == null)
            return;
        Camera cam = Futile.instance != null ? Futile.instance.camera : null;
        if (cam != null && cam.targetTexture == smallTarget)
        {
            cam.targetTexture = Futile.screen.renderTexture;
            cam.ResetAspect();
        }
        smallTarget.Release();
        UnityEngine.Object.Destroy(smallTarget);
        smallTarget = null;
    }

    /// <summary>
    /// Captures what the main camera rendered this frame, scaled to the target resolution.
    /// Call from the main camera's post-render callback.
    /// </summary>
    public byte[] CaptureFrame()
    {
        EnsureInitialized();

        RenderTexture originalActive = RenderTexture.active;

        try
        {
            Camera cam = Camera.main;
            RenderTexture source = cam != null ? cam.targetTexture : null;
            if (source != null)
                Downscale(source, renderTexture);
            else
                CaptureBackBuffer();

            // Read pixels from our scaled render texture. No Apply(): GetRawTextureData reads the CPU copy.
            RenderTexture.active = renderTexture;
            captureTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);

            // Get raw RGB data
            byte[] pixels = captureTexture.GetRawTextureData();

            return pixels;
        }
        finally
        {
            // Restore original active render texture
            RenderTexture.active = originalActive;
        }
    }

    /// <summary>
    /// Scales <paramref name="source"/> into <paramref name="dest"/> with bilinear filtering. A source that is a
    /// power-of-two multiple of the destination is halved step by step, so each output pixel is the exact
    /// average of its block.
    /// </summary>
    private static void Downscale(RenderTexture source, RenderTexture dest)
    {
        FilterMode filter = source.filterMode;
        source.filterMode = FilterMode.Bilinear;  // the game's own target is point-filtered
        RenderTexture current = source;
        while (current.width >= 4 * dest.width && current.height >= 4 * dest.height &&
               current.width % (2 * dest.width) == 0 && current.height % (2 * dest.height) == 0)
        {
            RenderTexture half = RenderTexture.GetTemporary(current.width / 2, current.height / 2, 0, RenderTextureFormat.ARGB32);
            half.filterMode = FilterMode.Bilinear;
            Graphics.Blit(current, half);
            if (current != source)
                RenderTexture.ReleaseTemporary(current);
            current = half;
        }
        Graphics.Blit(current, dest);
        if (current != source)
            RenderTexture.ReleaseTemporary(current);
        source.filterMode = filter;
    }

    /// <summary>Fallback for a camera that renders to the screen: copy the back buffer on the GPU and scale it.</summary>
    private void CaptureBackBuffer()
    {
        int screenWidth = Screen.width;
        int screenHeight = Screen.height;

        // Recreate screen capture texture if screen size changed
        if (screenCapture == null || screenWidth != lastScreenWidth || screenHeight != lastScreenHeight)
        {
            if (screenCapture != null)
            {
                screenCapture.Release();
                UnityEngine.Object.Destroy(screenCapture);
            }
            screenCapture = new RenderTexture(screenWidth, screenHeight, 0, RenderTextureFormat.ARGB32);
            screenCapture.Create();
            lastScreenWidth = screenWidth;
            lastScreenHeight = screenHeight;
        }

        // Copy the back buffer on the GPU (no full-size read back)
        RenderTexture.active = null;
        ScreenCapture.CaptureScreenshotIntoRenderTexture(screenCapture);

        // Blit the screen capture to our render texture (this scales it). The copy is upside down
        // where UVs start at the top (D3D, incl. under Wine), so flip it back to Unity's bottom-up rows.
        if (SystemInfo.graphicsUVStartsAtTop)
            Graphics.Blit(screenCapture, renderTexture, new Vector2(1f, -1f), new Vector2(0f, 1f));
        else
            Graphics.Blit(screenCapture, renderTexture);
    }

    /// <summary>
    /// Captures frame and flips it vertically (Unity textures are bottom-up).
    /// </summary>

    public byte[] CaptureFrameFlipped()
    {
        byte[] raw = CaptureFrame();
        byte[] flipped = new byte[raw.Length];

        int rowSize = width * 3;
        for (int y = 0; y < height; y++)
        {
            int srcRow = (height - 1 - y) * rowSize;
            int dstRow = y * rowSize;
            Array.Copy(raw, srcRow, flipped, dstRow, rowSize);
        }

        return flipped;
    }

    public void Dispose()
    {
        if (!disposed)
        {
            if (renderTexture != null)
            {
                renderTexture.Release();
                UnityEngine.Object.Destroy(renderTexture);
                renderTexture = null;
            }

            if (captureTexture != null)
            {
                UnityEngine.Object.Destroy(captureTexture);
                captureTexture = null;
            }

            if (screenCapture != null)
            {
                screenCapture.Release();
                UnityEngine.Object.Destroy(screenCapture);
                screenCapture = null;
            }

            ReleaseSmallTarget();

            disposed = true;
            initialized = false;
        }
    }
}
