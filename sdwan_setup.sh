#!/bin/bash

echo "[INFO] Configuring SD-WAN Mininet topology..."

# Enable IP forwarding on all routers
for r in r1 r2 r3; do
    mnexec -a $(pgrep -f "bash -il .* $r") sysctl -w net.ipv4.ip_forward=1
done

# --- R1 Configuration ---
mnexec -a $(pgrep -f "bash -il .* r1") ifconfig r1-eth0 7.7.7.2/24 up
mnexec -a $(pgrep -f "bash -il .* r1") ifconfig r1-eth1 10.0.0.1/24 up

# --- H1 Configuration (host-pc) ---
mnexec -a $(pgrep -f "bash -il .* h1") ifconfig h1-eth0 7.7.7.1/24 up
mnexec -a $(pgrep -f "bash -il .* h1") route add default gw 7.7.7.2

# --- H2 Configuration (PC1) ---
mnexec -a $(pgrep -f "bash -il .* h2") ifconfig h2-eth0 192.168.127.135/24 up
mnexec -a $(pgrep -f "bash -il .* h2") route add default gw 192.168.127.1

# --- R2 Configuration ---
mnexec -a $(pgrep -f "bash -il .* r2") ifconfig r2-eth0 10.0.1.1/24 up
mnexec -a $(pgrep -f "bash -il .* r2") ifconfig r2-eth1 172.168.1.1/24 up
mnexec -a $(pgrep -f "bash -il .* r2") ifconfig r2-eth2 172.168.2.1/24 up

# --- R3 Configuration ---
mnexec -a $(pgrep -f "bash -il .* r3") ifconfig r3-eth0 10.0.2.1/24 up
mnexec -a $(pgrep -f "bash -il .* r3") ifconfig r3-eth1 172.168.1.2/24 up
mnexec -a $(pgrep -f "bash -il .* r3") ifconfig r3-eth2 172.168.2.2/24 up

# --- Routes for R2 (to reach h2 via r3) ---
mnexec -a $(pgrep -f "bash -il .* r2") route add -net 192.168.127.0/24 gw 172.168.1.2
mnexec -a $(pgrep -f "bash -il .* r2") route add -net 192.168.127.0/24 gw 172.168.2.2

# --- Routes for R3 (to reach h1 via r2) ---
mnexec -a $(pgrep -f "bash -il .* r3") route add -net 7.7.7.0/24 gw 172.168.1.1
mnexec -a $(pgrep -f "bash -il .* r3") route add -net 7.7.7.0/24 gw 172.168.2.1

echo "[DONE] Topology configuration complete!"

