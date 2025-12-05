import { useState, useRef } from "react";
import axios from "axios";

export default function App() {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const resultRef = useRef(null);

  const analyze = async () => {
  if (!text || !image) {
    alert("Please enter text & upload an image!");
    return;
  }

  setLoading(true);

  const formData = new FormData();
  formData.append("text", text);
  formData.append("image", image);

  try {
    const res = await axios.post(
      "https://adarshbaalak-multimodal-analyzer.hf.space/proxy/7860/analyze",
      formData,
      { headers: { "Content-Type": "multipart/form-data" } }
    );

    setResult(res.data);

    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 300);

  } catch (error) {
    console.error(error);
    alert("Backend error");
  }

  setLoading(false);
};


  return (
    <div className="min-h-screen w-full flex flex-col items-center bg-black p-6 overflow-auto">

      <div className="w-full max-w-3xl bg-[#1a1a1a] text-white rounded-xl shadow-lg p-10 border border-[#292929]">

        <h1 className="text-3xl font-bold text-center mb-8">
          <span className="text-orange-500">Multimodal</span> Analyzer
        </h1>

        <label className="block text-sm mb-2 text-gray-300">Enter Text:</label>
        <textarea
          className="w-full p-4 mb-6 bg-[#111] text-white border border-[#333] rounded-lg focus:outline-none focus:border-orange-500"
          rows="4"
          placeholder="Type something..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        ></textarea>

        <label className="block text-sm mb-2 text-gray-300">Upload Image:</label>
        <input
          type="file"
          accept="image/*"
          className="mb-4"
          onChange={(e) => setImage(e.target.files[0])}
        />

        {image && (
          <img
            src={URL.createObjectURL(image)}
            className="w-full max-h-64 object-cover rounded-lg mt-3 border border-[#333]"
            alt="preview"
          />
        )}

        <button
          onClick={analyze}
          className="w-full mt-6 py-3 bg-orange-600 hover:bg-orange-700 text-white font-semibold rounded-lg transition-all"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {result && (
          <div
            ref={resultRef}
            className="mt-10 space-y-4 bg-[#111] p-6 rounded-lg border border-[#333]"
          >
            <h2 className="text-xl font-semibold text-orange-500 mb-4">Analysis Result</h2>

            <Result label="Sentiment" value={result.text_sentiment} />
            <Result label="Summary" value={result.text_summary} />
            <Result label="Topic" value={result.topic_classification} />
            <Result label="Image Classification" value={result.image_classification} />
            <Result label="OCR Text" value={result.ocr_text || "None"} />
            <Result label="Text Toxicity" value={result.text_toxicity_score} />
            <Result label="OCR Toxicity" value={result.ocr_toxicity_score} />

            <div className="p-4 bg-[#1e1e1e] border-l-4 border-orange-500 rounded-md">
              <p className="text-sm text-orange-400 font-semibold mb-1">Automated Response:</p>
              <p className="text-gray-300">{result.automated_response}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Result({ label, value }) {
  return (
    <div className="p-4 bg-[#1e1e1e] border border-[#333] rounded-lg">
      <p className="text-xs text-gray-400 uppercase tracking-wide">{label}</p>
      <p className="text-lg font-semibold text-white mt-1">{value}</p>
    </div>
  );
}
