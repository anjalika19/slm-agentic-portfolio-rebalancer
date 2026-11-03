// save as send.mjs, run: node send.mjs
import { ethers } from "ethers";

const RPC = "https://ethereum-sepolia-rpc.publicnode.com";
const KEY = "7e6fb1d5d4618e49b2675be7f614ba903ffa2719e512e7def86335da66a31818";       // export from MetaMask: account menu → Account details → Show private key

const provider = new ethers.JsonRpcProvider(RPC);
const wallet = new ethers.Wallet(KEY, provider);

const tx = await wallet.sendTransaction({
  to: wallet.address,
  value: 0,
  data: ethers.hexlify(ethers.toUtf8Bytes("krish-afk")),
});

console.log("Transaction hash:", tx.hash);
await tx.wait();
console.log("Mined.");