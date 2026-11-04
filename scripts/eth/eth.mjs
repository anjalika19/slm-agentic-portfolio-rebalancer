// Writes a short UTF-8 message into a zero-value Sepolia self-transaction.
//
//   cd scripts/eth && npm install
//   ETH_PRIVATE_KEY=0x... node eth.mjs ["message"]
//
// The private key is read from the environment and must never be committed.
import { ethers } from "ethers";

const RPC = process.env.ETH_RPC_URL || "https://ethereum-sepolia-rpc.publicnode.com";
const KEY = process.env.ETH_PRIVATE_KEY;
const MESSAGE = process.argv[2] || "krish-afk";

if (!KEY) {
  console.error("Set ETH_PRIVATE_KEY (and optionally ETH_RPC_URL) before running.");
  process.exit(1);
}

const provider = new ethers.JsonRpcProvider(RPC);
const wallet = new ethers.Wallet(KEY, provider);

const tx = await wallet.sendTransaction({
  to: wallet.address,
  value: 0,
  data: ethers.hexlify(ethers.toUtf8Bytes(MESSAGE)),
});

console.log("Transaction hash:", tx.hash);
await tx.wait();
console.log("Mined.");
