/**
 * Sealing a credential in the browser, so it is never sent in the clear.
 *
 * The flow this implements: fetch the master-node's public key, seal the SSH
 * key or password against it here, and send only the ciphertext. The REST API
 * stores what it receives and cannot open it — it holds no private key and its
 * environment has no crypto library at all. The credential is opened exactly
 * once, by the one process that runs simulations.
 *
 * Why libsodium and not WebCrypto
 * -------------------------------
 * Two reasons, and the first is decisive:
 *
 * 1. **WebCrypto has no sealed box.** It offers ECDH and AES-GCM, from which
 *    we would have to *design* an envelope and write the matching Python half
 *    by hand. `crypto_box_seal` is a standard, audited construction with a
 *    PyNaCl counterpart that opens it byte for byte — verified by a fixed
 *    vector in this module's test and in `pylib/tests/test_sealed_secrets.py`.
 * 2. **`crypto.subtle` requires a secure context.** The GUI is reachable over
 *    plain HTTP on a LAN address, where `window.crypto.subtle` is simply
 *    `undefined`. A sealing path that silently stopped existing outside
 *    HTTPS is not a path worth having, even now that the stack ships a TLS
 *    proxy.
 *
 * The WASM module is ~190 kB, so it is imported dynamically: nothing is
 * downloaded until someone actually opens the host form.
 */
import type { PublicKeyDto, SealedSecret } from "../types/infra";

/** The only algorithm this codebase accepts. */
export const ENVELOPE_ALG = "crypto_box_seal";

/**
 * A sealed box is a 32-byte ephemeral public key plus a 16-byte MAC around the
 * plaintext, so 48 bytes is the floor even for an empty secret. Mirrors
 * `pylib/crypto/envelope.py`, which refuses anything shorter — the realistic
 * mistake being a password base64'd by hand.
 */
export const SEALED_BOX_OVERHEAD = 48;

/** An ed25519 key is ~400 bytes of PEM, an RSA-4096 one ~3.2 kB. */
export const MAX_PLAINTEXT_BYTES = 16 * 1024;

export class SealingError extends Error {}

type Sodium = {
  ready: Promise<void>;
  crypto_box_seal: (message: Uint8Array, publicKey: Uint8Array) => Uint8Array;
  from_base64: (input: string, variant: number) => Uint8Array;
  to_base64: (input: Uint8Array, variant: number) => string;
  from_string: (input: string) => Uint8Array;
  base64_variants: { ORIGINAL: number };
};

let sodiumPromise: Promise<Sodium> | null = null;

/** Load libsodium once, on first use. */
async function loadSodium(): Promise<Sodium> {
  if (sodiumPromise === null) {
    sodiumPromise = (async () => {
      const module = await import("libsodium-wrappers");
      const sodium = (module.default ?? module) as unknown as Sodium;
      await sodium.ready;
      return sodium;
    })().catch((error) => {
      // Do not cache a failure: a transient network problem on the first
      // open should not make the form permanently unusable.
      sodiumPromise = null;
      throw new SealingError(
        `Could not load the encryption library: ${String(error)}`,
      );
    });
  }
  return sodiumPromise;
}

/**
 * Seal `plaintext` against the master-node's public key.
 *
 * The plaintext exists only as an argument here and is never stored, logged
 * or returned.
 */
export async function sealSecret(
  plaintext: string,
  publicKey: Pick<PublicKeyDto, "key_id" | "public_key" | "alg">,
): Promise<SealedSecret> {
  if (!plaintext) {
    throw new SealingError("Nothing to seal: the credential is empty.");
  }
  const bytes = new TextEncoder().encode(plaintext);
  if (bytes.length > MAX_PLAINTEXT_BYTES) {
    throw new SealingError(
      `The credential is larger than ${MAX_PLAINTEXT_BYTES} bytes.`,
    );
  }
  if (publicKey.alg && publicKey.alg !== ENVELOPE_ALG) {
    throw new SealingError(
      `The master-node published a key for ${publicKey.alg}, which this ` +
        `client cannot seal against.`,
    );
  }
  if (!publicKey.public_key || !publicKey.key_id) {
    throw new SealingError("The master-node has not published a public key.");
  }

  const sodium = await loadSodium();
  const variant = sodium.base64_variants.ORIGINAL;
  let recipient: Uint8Array;
  try {
    recipient = sodium.from_base64(publicKey.public_key, variant);
  } catch {
    throw new SealingError("The published public key is not valid base64.");
  }

  const ciphertext = sodium.to_base64(
    sodium.crypto_box_seal(bytes, recipient),
    variant,
  );
  return { key_id: publicKey.key_id, ciphertext };
}

/**
 * Whether a value could be a sealed envelope.
 *
 * The same arithmetic the server applies, repeated here so the form can say
 * "that is not sealed" before anything is sent rather than surfacing a 422.
 */
export function looksSealed(secret: unknown): secret is SealedSecret {
  if (typeof secret !== "object" || secret === null) return false;
  const candidate = secret as Partial<SealedSecret>;
  if (!candidate.key_id || !candidate.ciphertext) return false;
  try {
    const raw = atob(candidate.ciphertext);
    return raw.length > SEALED_BOX_OVERHEAD;
  } catch {
    return false;
  }
}

/**
 * A fingerprint of the *key* a secret was sealed against, for display.
 *
 * Not of the secret: nothing here can read it, and showing a digest of the
 * ciphertext would only invite someone to treat it as an identity.
 */
export function describeSealed(
  secret: SealedSecret,
  publicKey?: Pick<PublicKeyDto, "key_id" | "fingerprint">,
): string {
  if (publicKey && publicKey.key_id === secret.key_id) {
    return `sealed against ${publicKey.fingerprint}`;
  }
  return `sealed against key ${secret.key_id}`;
}
