import './App.css';
import { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function segment() {
    if (!loading && file) {
      try {
        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(
          "http://127.0.0.1:8000/segment",
          {
            method: "POST",
            body: formData,
          }
        );

        if (!response.ok) {
          throw new Error("Segmentation failed");
        }

        const data = await response.json();

        setResult(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
  }

  return (
    <div className="app">

      <header className="header">
        <h1>Brain & Breast Tumor Segmentation</h1>
        <p>
          AI-powered medical image segmentation
        </p>
      </header>

      <main className="container">

        {/* Upload section */}
        <section className="upload-section">

          <h2>Upload Medical Image</h2>

          <div className="upload-box">

            <input
              id="image-upload"
              type="file"
              accept="image/png,image/jpeg"
              onChange={(event) => {
                const selectedFile = event.target.files[0];

                if (!selectedFile) return;

                setFile(selectedFile);
                setError(null);
                setResult(null);

                setPreview(
                  URL.createObjectURL(selectedFile)
                );
              }}
            />

            <label htmlFor="image-upload">
              Choose an image
            </label>

            {file && (
              <p className="file-name">
                {file.name}
              </p>
            )}

          </div>

          <button
            className="segment-button"
            onClick={segment}
            disabled={loading || !file}
          >
            {loading
              ? "Segmenting..."
              : "Segment Image"}
          </button>

        </section>


        {/* Results section */}
        <section className="results-section">

          {preview && (
            <div className="original-card image-card">
              <div className="card-header">
                <h2>Original Image</h2>
                <span>512 × 512</span>
              </div>

              <div className="image-container">
                <img
                  src={preview}
                  alt="Original medical image"
                />
              </div>
            </div>
          )}

          {result && (
            <div className="segmentation-section">

              <div className="section-title">
                <h2>Segmentation Results</h2>
                <p>
                  Comparison of tumor segmentation produced by the three models.
                </p>
              </div>

              <div className="results-grid">

                <div className="image-card">
                  <div className="card-header">
                    <h3>U-Net</h3>
                    <span>512 × 512</span>
                  </div>

                  <div className="image-container">
                    <img
                      src={`data:image/png;base64,${result.masks.unet}`}
                      alt="U-Net segmentation"
                    />
                  </div>
                </div>


                <div className="image-card">
                  <div className="card-header">
                    <h3>U-Net++</h3>
                    <span>512 × 512</span>
                  </div>

                  <div className="image-container">
                    <img
                      src={`data:image/png;base64,${result.masks["unet++"]}`}
                      alt="U-Net++ segmentation"
                    />
                  </div>
                </div>


                <div className="image-card">
                  <div className="card-header">
                    <h3>DeepLabV3</h3>
                    <span>512 × 512</span>
                  </div>

                  <div className="image-container">
                    <img
                      src={`data:image/png;base64,${result.masks.deeplabv3}`}
                      alt="DeepLabV3 segmentation"
                    />
                  </div>
                </div>

              </div>

            </div>
          )}

        </section>

        {/* Error */}
        {error && (
          <div className="error">
            {error}
          </div>
        )}

      </main>

    </div>
  );
}

export default App;
