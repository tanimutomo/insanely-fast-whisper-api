const {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
} = require("@aws-sdk/client-s3");
const {
  TranscribeClient,
  StartTranscriptionJobCommand,
  GetTranscriptionJobCommand,
  DeleteTranscriptionJobCommand,
} = require("@aws-sdk/client-transcribe");
const { awsCredentialsProvider } = require("@vercel/oidc-aws-credentials-provider");
const crypto = require("crypto");

const REGION = (process.env.AWS_REGION || "ap-northeast-1").trim();
const ROLE_ARN = (process.env.AWS_ROLE_ARN || "").trim();
const BUCKET = (process.env.TRANSCRIBE_S3_BUCKET || "").trim();
const VOCABULARY = (process.env.TRANSCRIBE_VOCABULARY_NAME || "medical-terms-ja").trim();

const credentials = awsCredentialsProvider({ roleArn: ROLE_ARN });
const s3 = new S3Client({ region: REGION, credentials });
const transcribe = new TranscribeClient({ region: REGION, credentials });

const EXT_BY_MIME = {
  "audio/webm": "webm",
  "audio/ogg": "ogg",
  "audio/mp4": "mp4",
  "audio/m4a": "m4a",
  "audio/x-m4a": "m4a",
  "audio/mpeg": "mp3",
  "audio/mp3": "mp3",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/flac": "flac",
};

async function readRawBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, X-Language");

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  if (!ROLE_ARN || !BUCKET) {
    return res.status(500).json({ error: "Missing AWS_ROLE_ARN or TRANSCRIBE_S3_BUCKET env var" });
  }

  const contentType = (req.headers["content-type"] || "application/octet-stream").split(";")[0].trim();
  const language = (req.headers["x-language"] || "ja-JP").toString();
  const ext = EXT_BY_MIME[contentType] || "webm";

  const jobId = `whisper-compare-${Date.now()}-${crypto.randomBytes(4).toString("hex")}`;
  const s3Key = `uploads/${jobId}.${ext}`;
  const mediaUri = `s3://${BUCKET}/${s3Key}`;

  try {
    const start = Date.now();

    const body = await readRawBody(req);
    if (body.length === 0) return res.status(400).json({ error: "Empty audio body" });

    await s3.send(new PutObjectCommand({
      Bucket: BUCKET,
      Key: s3Key,
      Body: body,
      ContentType: contentType,
    }));

    await transcribe.send(new StartTranscriptionJobCommand({
      TranscriptionJobName: jobId,
      LanguageCode: language,
      Media: { MediaFileUri: mediaUri },
      Settings: { VocabularyName: VOCABULARY },
    }));

    // Poll until COMPLETED or FAILED. Stay well under the 300s function cap.
    const deadline = Date.now() + 270_000;
    let transcriptUri = null;
    let failureReason = null;
    while (Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 3000));
      const { TranscriptionJob } = await transcribe.send(
        new GetTranscriptionJobCommand({ TranscriptionJobName: jobId })
      );
      const status = TranscriptionJob.TranscriptionJobStatus;
      if (status === "COMPLETED") {
        transcriptUri = TranscriptionJob.Transcript.TranscriptFileUri;
        break;
      }
      if (status === "FAILED") {
        failureReason = TranscriptionJob.FailureReason;
        break;
      }
    }

    // Best-effort cleanup regardless of outcome
    const cleanup = async () => {
      await Promise.allSettled([
        s3.send(new DeleteObjectCommand({ Bucket: BUCKET, Key: s3Key })),
        transcribe.send(new DeleteTranscriptionJobCommand({ TranscriptionJobName: jobId })),
      ]);
    };

    if (failureReason) {
      await cleanup();
      return res.status(502).json({ error: `Transcribe failed: ${failureReason}` });
    }
    if (!transcriptUri) {
      await cleanup();
      return res.status(504).json({ error: "Transcribe timed out" });
    }

    const transcriptResp = await fetch(transcriptUri);
    if (!transcriptResp.ok) {
      await cleanup();
      return res.status(502).json({ error: `Fetch transcript failed: ${transcriptResp.status}` });
    }
    const transcript = await transcriptResp.json();
    const text = (transcript.results?.transcripts || [])
      .map(t => t.transcript)
      .join(" ")
      .trim();

    await cleanup();

    const elapsed = (Date.now() - start) / 1000;
    return res.status(200).json({
      text,
      processing_time_s: Math.round(elapsed * 100) / 100,
      vocabulary: VOCABULARY,
    });
  } catch (err) {
    console.error("transcribe-aws error", err);
    return res.status(500).json({ error: err.message || String(err) });
  }
};

module.exports.config = {
  maxDuration: 300,
};
