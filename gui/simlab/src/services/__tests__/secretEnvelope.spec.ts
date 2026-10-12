/**
 * Cross-language compatibility is the thing being pinned here.
 *
 * The browser seals and the master-node opens, with no shared code between
 * them, so the only guarantee is that both speak `crypto_box_seal`. The fixed
 * keypair below is the same one `pylib/tests/test_sealed_secrets.py` uses; a
 * ciphertext produced here opens there, and that file holds ciphertexts
 * produced here that must keep opening.
 *
 * If this file and that one ever disagree, registering a host stops working —
 * which is exactly the failure this pair of tests exists to catch early.
 */
import { beforeAll, describe, expect, it } from "vitest";
import _sodium from "libsodium-wrappers";
import {
  ENVELOPE_ALG,
  MAX_PLAINTEXT_BYTES,
  SealingError,
  describeSealed,
  looksSealed,
  sealSecret,
} from "../secretEnvelope";

// The throwaway keypair shared with the Python test. Test-only, by design.
const VECTOR_PRIVATE_KEY = "Ip31WMHfEDvJ8raxkABwYNJOXxjTFUEhWtC79++uXHs=";
const VECTOR_PUBLIC_KEY = "dowFp5nE7ZgT6hoHVOwOPOfnOtOiRg+ONJ4+ewJ+N04=";
const VECTOR_KEY_ID = "b114dd6d8410a62a";

const PUBLIC_KEY = {
  key_id: VECTOR_KEY_ID,
  public_key: VECTOR_PUBLIC_KEY,
  alg: ENVELOPE_ALG,
  fingerprint: "SHA256:test",
};

let sodium: typeof _sodium;

beforeAll(async () => {
  await _sodium.ready;
  sodium = _sodium;
});

/** Open an envelope with the vector private key, as the master-node would. */
function openWithVectorKey(ciphertext: string): string {
  const variant = sodium.base64_variants.ORIGINAL;
  return sodium.to_string(
    sodium.crypto_box_seal_open(
      sodium.from_base64(ciphertext, variant),
      sodium.from_base64(VECTOR_PUBLIC_KEY, variant),
      sodium.from_base64(VECTOR_PRIVATE_KEY, variant),
    ),
  );
}

describe("sealing a credential", () => {
  it("produces something only the private key can open", async () => {
    const envelope = await sealSecret("hunter2", PUBLIC_KEY);
    expect(envelope.key_id).toBe(VECTOR_KEY_ID);
    expect(openWithVectorKey(envelope.ciphertext)).toBe("hunter2");
  });

  it.each([
    ["an OpenSSH private key", "-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blbg==\n-----END OPENSSH PRIVATE KEY-----"],
    ["a password with accents", "senha com acento: ção"],
    ["a single character", "x"],
    ["the maximum size", "a".repeat(MAX_PLAINTEXT_BYTES)],
  ])("round-trips %s", async (_label, plaintext) => {
    const envelope = await sealSecret(plaintext, PUBLIC_KEY);
    expect(openWithVectorKey(envelope.ciphertext)).toBe(plaintext);
  });

  it("never produces the same ciphertext twice", async () => {
    // Sealed boxes use a fresh ephemeral key per call, so two hosts sharing a
    // credential must not produce comparable values.
    const first = await sealSecret("same", PUBLIC_KEY);
    const second = await sealSecret("same", PUBLIC_KEY);
    expect(first.ciphertext).not.toBe(second.ciphertext);
  });

  it("refuses an empty credential", async () => {
    await expect(sealSecret("", PUBLIC_KEY)).rejects.toThrow(SealingError);
  });

  it("refuses one that is too large", async () => {
    await expect(
      sealSecret("a".repeat(MAX_PLAINTEXT_BYTES + 1), PUBLIC_KEY),
    ).rejects.toThrow(/larger than/);
  });

  it("refuses a key published for another algorithm", async () => {
    // Better than silently sealing with something the other side cannot open.
    await expect(
      sealSecret("x", { ...PUBLIC_KEY, alg: "aes-gcm" }),
    ).rejects.toThrow(/cannot seal against/);
  });

  it("refuses a missing public key", async () => {
    await expect(
      sealSecret("x", { ...PUBLIC_KEY, public_key: "" }),
    ).rejects.toThrow(/has not published/);
  });

  it("refuses a public key that is not base64", async () => {
    await expect(
      sealSecret("x", { ...PUBLIC_KEY, public_key: "not base64!!" }),
    ).rejects.toThrow(/not valid base64/);
  });
});

describe("recognising a sealed envelope", () => {
  it("accepts what sealSecret produced", async () => {
    expect(looksSealed(await sealSecret("hunter2", PUBLIC_KEY))).toBe(true);
  });

  it("rejects a password base64'd by hand", () => {
    // The realistic mistake. Valid base64, but far too short to be a sealed
    // box — the same arithmetic the server applies, so the form can say so
    // before a request is made.
    expect(looksSealed({ key_id: "0".repeat(16), ciphertext: btoa("hunter2") }))
      .toBe(false);
  });

  it.each([
    ["a bare string", "hunter2"],
    ["null", null],
    ["a missing ciphertext", { key_id: "0".repeat(16) }],
    ["a missing key id", { ciphertext: "x".repeat(100) }],
    ["invalid base64", { key_id: "0".repeat(16), ciphertext: "not base64!!" }],
  ])("rejects %s", (_label, value) => {
    expect(looksSealed(value)).toBe(false);
  });
});

describe("describing a sealed envelope", () => {
  it("names the key's fingerprint when it is the active one", async () => {
    const envelope = await sealSecret("x", PUBLIC_KEY);
    expect(describeSealed(envelope, PUBLIC_KEY)).toBe(
      "sealed against SHA256:test",
    );
  });

  it("falls back to the key id for a retired key", async () => {
    const envelope = await sealSecret("x", PUBLIC_KEY);
    expect(
      describeSealed(envelope, { key_id: "other", fingerprint: "SHA256:other" }),
    ).toBe(`sealed against key ${VECTOR_KEY_ID}`);
  });
});
