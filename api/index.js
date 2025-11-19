import { InferenceSession, Tensor } from 'onnxruntime-web';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), { status: 405 });
  }

  try {
    const body = await req.json();
    const { TransactionAmount, AccountBalance } = body;

    if (TransactionAmount == null || AccountBalance == null) {
      return new Response(JSON.stringify({ error: 'TransactionAmount and AccountBalance required' }), { status: 400 });
    }

    // Convert input ke float32 karena onnxruntime-web tidak support float64
    const inputTensor = new Tensor('float32', Float32Array.from([TransactionAmount, AccountBalance]), [1, 2]);

    // Path model
    const modelPath = path.join(__dirname, '../model/model.onnx');

    // Load session ONNX
    const session = await InferenceSession.create(modelPath);

    // Jalankan inference
    const results = await session.run({ input: inputTensor });

    return new Response(JSON.stringify({ output: results.output.data }), { status: 200 });

  } catch (err) {
    console.error(err);
    return new Response(JSON.stringify({ error: err.message }), { status: 500 });
  }
}
