import fs from "node:fs/promises";

const INPUT = "src/data/certifications-sources.json";
const OUTPUT = "src/data/certifications.json";

function getMeta(html, key) {
  const patterns = [
    new RegExp(`<meta[^>]+property=["']${key}["'][^>]+content=["']([^"']+)["']`, "i"),
    new RegExp(`<meta[^>]+content=["']([^"']+)["'][^>]+property=["']${key}["']`, "i"),
    new RegExp(`<meta[^>]+name=["']${key}["'][^>]+content=["']([^"']+)["']`, "i"),
    new RegExp(`<meta[^>]+content=["']([^"']+)["'][^>]+name=["']${key}["']`, "i")
  ];

  for (const pattern of patterns) {
    const match = html.match(pattern);
    if (match?.[1]) return decodeHtml(match[1].trim());
  }

  return null;
}

function getTitleTag(html) {
  const match = html.match(/<title[^>]*>(.*?)<\/title>/is);
  return match?.[1] ? decodeHtml(match[1].replace(/\s+/g, " ").trim()) : null;
}

function decodeHtml(value = "") {
  return value
    .replaceAll("&amp;", "&")
    .replaceAll("&quot;", "\"")
    .replaceAll("&#39;", "'")
    .replaceAll("&apos;", "'")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">");
}

function cleanTitle(title, fallbackTitle) {
  if (!title) return fallbackTitle || "Verified Credential";

  let cleaned = title
    .replace(/\s*-\s*Credly\s*$/i, "")
    .replace(/\s*\|\s*Coursera\s*$/i, "")
    .replace(/\s*Coursera\s*$/i, "")
    .replace(/\s+/g, " ")
    .trim();

  // Credly sometimes returns: "X was issued by Coursera to Luis..."
  const issuedMatch = cleaned.match(/^(.+?)\s+was issued by\s+.+?\s+to\s+.+\.?$/i);
  if (issuedMatch?.[1]) {
    cleaned = issuedMatch[1].trim();
  }

  // Avoid useless generic titles
  if (!cleaned || /^credly$/i.test(cleaned) || /^badge wallet$/i.test(cleaned)) {
    return fallbackTitle || "Verified Credential";
  }

  return cleaned;
}

function detectPlatform(url) {
  if (url.includes("credly.com")) return "Credly";
  if (url.includes("coursera.org")) return "Coursera";
  return "Credential";
}

async function readJsonSafe(path, fallbackValue) {
  try {
    const raw = await fs.readFile(path, "utf8");
    return JSON.parse(raw);
  } catch {
    return fallbackValue;
  }
}

async function fetchMetadata(source) {
  try {
    const response = await fetch(source.url, {
      headers: {
        "User-Agent": "Mozilla/5.0 Portfolio Certification Metadata Bot"
      }
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const html = await response.text();

    const ogTitle = getMeta(html, "og:title");
    const titleTag = getTitleTag(html);

    const scrapedTitle = cleanTitle(
      ogTitle || titleTag,
      source.fallbackTitle || source.title
    );

    const description =
      getMeta(html, "og:description") ||
      getMeta(html, "description") ||
      source.description ||
      "";

    const image =
      getMeta(html, "og:image") ||
      getMeta(html, "twitter:image") ||
      source.image ||
      "";

    return {
      // IMPORTANT:
      // Manual title wins. Credly scraped titles are only fallback.
      title: source.fallbackTitle || source.title || scrapedTitle,

      issuer: source.issuer,
      platform: source.platform || detectPlatform(source.url),
      date: source.date,
      credentialId: source.credentialId,
      url: source.url,
      category: source.category,
      featured: Boolean(source.featured),
      skills: source.skills || [],
      image,
      description
    };
  } catch (error) {
    console.warn(`Metadata fetch failed for ${source.url}: ${error.message}`);

    return {
      title: source.fallbackTitle || source.title || "Verified Credential",
      issuer: source.issuer,
      platform: source.platform || detectPlatform(source.url),
      date: source.date,
      credentialId: source.credentialId,
      url: source.url,
      category: source.category,
      featured: Boolean(source.featured),
      skills: source.skills || [],
      image: source.image || "",
      description: source.description || ""
    };
  }
}

async function main() {
  const sources = await readJsonSafe(INPUT, []);
  const existing = await readJsonSafe(OUTPUT, []);

  const byUrl = new Map();

  for (const cert of existing) {
    if (cert.url) byUrl.set(cert.url, cert);
  }

  for (const source of sources) {
    const updated = await fetchMetadata(source);
    const previous = byUrl.get(source.url) || {};

    byUrl.set(source.url, {
      ...previous,
      ...updated,
      skills: updated.skills?.length ? updated.skills : previous.skills || [],
      image: updated.image || previous.image || "",
      description: updated.description || previous.description || ""
    });
  }

  const certifications = Array.from(byUrl.values());

  certifications.sort((a, b) => {
    const aFeatured = a.featured ? 1 : 0;
    const bFeatured = b.featured ? 1 : 0;

    if (bFeatured !== aFeatured) return bFeatured - aFeatured;

    return String(b.date || "").localeCompare(String(a.date || ""));
  });

  await fs.writeFile(OUTPUT, JSON.stringify(certifications, null, 2) + "\n", "utf8");

  console.log(`Updated ${OUTPUT}`);
  console.log(`Existing certifications preserved: ${existing.length}`);
  console.log(`Sources processed: ${sources.length}`);
  console.log(`Total certifications: ${certifications.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});