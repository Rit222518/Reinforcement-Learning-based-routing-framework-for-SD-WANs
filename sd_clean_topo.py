#!/usr/bin/python3

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.node import RemoteController
from mininet.log import setLogLevel, info


class CleanSDWANTopo(Topo):
    def build(self):
        # Hosts
        h1 = self.addHost('h1', ip='10.0.1.1/24')
        h2 = self.addHost('h2', ip='10.0.2.1/24')

        # Routers
        r1 = self.addHost('r1')
        r2 = self.addHost('r2')

        # Switches
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # Host to switch links
        self.addLink(h1, s1)
        self.addLink(h2, s2)

        # Router to switch links
        self.addLink(r1, s1)  # r1-eth0 <-> s1
        self.addLink(r2, s2)  # r2-eth0 <-> s2

        # Inter-router link
        self.addLink(r1, r2)  # r1-eth1 <-> r2-eth1


def run():
    topo = CleanSDWANTopo()

    controller_ip = '127.0.0.1'
    controller_port = 6653

    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip=controller_ip, port=controller_port),
        link=TCLink,
        autoSetMacs=True
    )

    info('[INFO] Starting network...\n')
    net.start()

    info('[INFO] Enabling IP forwarding on routers...\n')
    net.get('r1').cmd('sysctl -w net.ipv4.ip_forward=1')
    net.get('r2').cmd('sysctl -w net.ipv4.ip_forward=1')

    info('[INFO] Configuring router interfaces...\n')
    r1 = net.get('r1')
    r2 = net.get('r2')

    r1.cmd('ifconfig r1-eth0 10.0.1.254/24')
    r1.cmd('ifconfig r1-eth1 192.168.1.1/24')

    r2.cmd('ifconfig r2-eth0 10.0.2.254/24')
    r2.cmd('ifconfig r2-eth1 192.168.1.2/24')

    info('[INFO] Setting host default gateways...\n')
    h1 = net.get('h1')
    h2 = net.get('h2')
    h1.cmd('ip route add default via 10.0.1.254')
    h2.cmd('ip route add default via 10.0.2.254')

    info('[INFO] Adding static routes to routers...\n')
    r1.cmd('ip route add 10.0.2.0/24 via 192.168.1.2')
    r2.cmd('ip route add 10.0.1.0/24 via 192.168.1.1')

    info('[INFO] Running CLI...\n')
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run()
