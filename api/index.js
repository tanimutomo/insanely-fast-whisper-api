const fs = require("fs");
const path = require("path");

module.exports = (req, res) => {
  const user = process.env.BASIC_AUTH_USER || "admin";
  const pass = process.env.BASIC_AUTH_PASSWORD || "password";

  const auth = req.headers.authorization;
  if (auth) {
    const [scheme, encoded] = auth.split(" ");
    if (scheme === "Basic" && encoded) {
      const decoded = Buffer.from(encoded, "base64").toString("utf-8");
      const [u, p] = decoded.split(":");
      if (u === user && p === pass) {
        const page = req.url === "/compare" ? "compare.html" : "demo.html";
        const html = fs.readFileSync(
          path.join(__dirname, "..", page),
          "utf-8"
        );
        res.setHeader("Content-Type", "text/html");
        return res.status(200).send(html);
      }
    }
  }

  res.setHeader("WWW-Authenticate", 'Basic realm="Secure Area"');
  return res.status(401).send("Authentication required");
};
