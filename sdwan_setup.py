def configure_sdwan(net):
    print("[INFO] Configuring SD-WAN topology...")

    # --- Enable IP forwarding on routers ---
    for r in ['r1', 'r2', 'r3']:
        router = net.get(r)
        router.cmd("sysctl -w net.ipv4.ip_forward=1")

    # --- h1 <-> r1 ---
    h1 = net.get('h1')
    r1 = net.get('r1')
    h1.cmd("ifconfig h1-eth0 7.7.7.1/24 up")
    h1.cmd("route add default gw 7.7.7.2")
    r1.cmd("ifconfig r1-eth0 7.7.7.2/24 up")
    r1.cmd("ifconfig r1-eth1 10.0.0.1/24 up")

    # --- h2 <-> r3 ---
    h2 = net.get('h2')
    r3 = net.get('r3')
    h2.cmd("ifconfig h2-eth0 192.168.127.135/24 up")
    h2.cmd("route add default gw 192.168.127.1")
    r3.cmd("ifconfig r3-eth0 10.0.2.1/24 up")
    r3.cmd("ifconfig r3-eth1 172.168.1.2/24 up")
    r3.cmd("ifconfig r3-eth2 172.168.2.2/24 up")
    r3.cmd("ifconfig r3-eth3 192.168.127.1/24 up")

    # --- r2 configuration ---
    r2 = net.get('r2')
    r2.cmd("ifconfig r2-eth0 10.0.1.1/24 up")
    r2.cmd("ifconfig r2-eth1 172.168.1.1/24 up")
    r2.cmd("ifconfig r2-eth2 172.168.2.1/24 up")

    # --- Routing rules ---
    r2.cmd("route add -net 192.168.127.0/24 gw 172.168.1.2")
    r2.cmd("route add -net 192.168.127.0/24 gw 172.168.2.2")

    r3.cmd("route add -net 7.7.7.0/24 gw 172.168.1.1")
    r3.cmd("route add -net 7.7.7.0/24 gw 172.168.2.1")

    print("[DONE] SD-WAN setup complete.")
