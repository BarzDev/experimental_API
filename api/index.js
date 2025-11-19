import * as ort from "onnxruntime-web";

let sessionPromise = null;

export default async function handler(req, res) {
  try {
    if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

    if (!sessionPromise) {
      sessionPromise = ort.InferenceSession.create(new URL("./model.onnx", import.meta.url).href, { executionProviders: ["wasm"] });
    }
    const session = await sessionPromise;

    const { TransactionAmount, AccountBalance } = req.body;

    const inputTensor = new ort.Tensor("float32", new Float32Array([TransactionAmount, AccountBalance]), [1, 2]);
    const results = await session.run({ input: inputTensor });
    const output = results.output.data[0];

    res.status(200).json({ cluster: output });
  } catch (err) {
    res.status(500).json({ error: err.toString() });
  }
}
