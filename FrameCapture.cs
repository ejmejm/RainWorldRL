using System;
using UnityEngine;

/// <summary>
/// Captures game frames at a configurable resolution for RL observation.
/// Uses Unity's RenderTexture to scale the game camera output.
/// Uses lazy initialization to avoid creating textures before Unity is ready.
/// </summary>
public class FrameCapture : IDisposable
{
    private RenderTexture renderTexture;
    private Texture2D captureTexture;
    private Texture2D screenCapture;  // Cached for reading from screen
    private int width;
    private int height;
    private int lastScreenWidth;
    private int lastScreenHeight;
    private bool disposed = false;
    private bool initialized = false;

    public int Width => width;
    public int Height => height;
    public int FrameSize => width * height * 3;

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
    }

    /// <summary>
    /// Captures the current frame from what's already rendered on screen.
    /// Should be called after rendering is complete (e.g., in OnPostRender or after WaitForEndOfFrame).
    /// Reads from screen and scales to target resolution.
    /// </summary>
    public byte[] CaptureFrame()
    {
        EnsureInitialized();

        RenderTexture originalActive = RenderTexture.active;

        try
        {
            int screenWidth = Screen.width;
            int screenHeight = Screen.height;

            // Recreate screen capture texture if screen size changed
            if (screenCapture == null || screenWidth != lastScreenWidth || screenHeight != lastScreenHeight)
            {
                if (screenCapture != null)
                {
                    UnityEngine.Object.Destroy(screenCapture);
                }
                screenCapture = new Texture2D(screenWidth, screenHeight, TextureFormat.RGB24, false);
                lastScreenWidth = screenWidth;
                lastScreenHeight = screenHeight;
            }

            // IMPORTANT: Set active RT to null to read from the screen back buffer
            RenderTexture.active = null;

            // Read directly from screen into our cached texture
            screenCapture.ReadPixels(new Rect(0, 0, screenWidth, screenHeight), 0, 0);
            screenCapture.Apply();

            // Blit the screen capture to our render texture (this scales it)
            Graphics.Blit(screenCapture, renderTexture);

            // Read pixels from our scaled render texture
            RenderTexture.active = renderTexture;
            captureTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            captureTexture.Apply();

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
                UnityEngine.Object.Destroy(screenCapture);
                screenCapture = null;
            }

            disposed = true;
            initialized = false;
        }
    }
}
