const { awsCredentialsProvider } = require("@vercel/oidc-aws-credentials-provider");

// Trim whitespace defensively — env values set via `echo | vercel env add`
// can carry a trailing newline that breaks the AWS SDK's hostname builder.
const ROLE_ARN = (process.env.AWS_ROLE_ARN || "").trim();
const REGION = (process.env.AWS_REGION || "ap-northeast-1").trim();
const VOCABULARY = (process.env.TRANSCRIBE_VOCABULARY_NAME || "medical-terms-ja").trim();

// Short-lived credentials for browser-side Amazon Transcribe Streaming.
// The IAM role trust policy restricts to Vercel OIDC for this project, and
// the role's inline policy restricts what these credentials can do. Still,
// we expose them for 15 minutes only and only for scoped streaming use.
// Default session is 1 hour via AssumeRoleWithWebIdentity. Scope of these
// credentials is constrained by the IAM role policy (S3 uploads + Transcribe),
// so short TTL is less critical than tight permissions.
const credentials = awsCredentialsProvider({ roleArn: ROLE_ARN });

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "GET") return res.status(405).json({ error: "Method not allowed" });

  if (!ROLE_ARN) {
    return res.status(500).json({ error: "AWS_ROLE_ARN not configured" });
  }

  try {
    const creds = await credentials();
    return res.status(200).json({
      accessKeyId: creds.accessKeyId,
      secretAccessKey: creds.secretAccessKey,
      sessionToken: creds.sessionToken,
      expiration: creds.expiration,
      region: REGION,
      vocabularyName: VOCABULARY,
    });
  } catch (err) {
    console.error("aws-credentials error", err);
    return res.status(500).json({ error: err.message || String(err) });
  }
};
