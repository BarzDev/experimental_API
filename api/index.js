const ort = require('onnxruntime-node');
const path = require('path');

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { TransactionAmount, AccountBalance } = req.body;

    if (TransactionAmount == null || AccountBalance == null) {
      return res.status(400).json({ error: 'TransactionAmount and AccountBalance required' });
    }

    // Buat tensor input (float64 untuk KMeans / scikit-learn ONNX)
    const inputTensor = new ort.Tensor(
      'float64',
      Float64Array.from([TransactionAmount, AccountBalance]),
      [1, 2]  // shape: [1, 2]
    );

    // Path model
    const modelPath = path.join(__dirname, '../model/model.onnx');

    // Load session ONNX
    const session = await ort.InferenceSession.create(modelPath);

    // Jalankan inference
    const feeds = { input: inputTensor }; // nama input harus sama dengan model
    const results = await session.run(feeds);

    res.status(200).json({ output: results.output.data });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
};
