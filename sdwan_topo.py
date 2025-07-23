from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from sdwan_setup import configure_sdwan  # 🧠 Your custom IP/routing setup

class SDWANTopo(Topo):
    def build(self):
        # Hosts and Routers
        h1 = self.addHost('h1')  # host-pc
        h2 = self.addHost('h2')  # pc1
        h3 = self.addHost('h3')  # pc2 (optional)
        h4 = self.addHost('h4')  # pc3 (optional)

        r1 = self.addHost('r1')  # Edge router 1
        r2 = self.addHost('r2')  # Core router 2
        r3 = self.addHost('r3')  # Core router 3

        # Switches
        s0 = self.addSwitch('s0')
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        # Host-Router Links
        self.addLink(h1, r1)
        self.addLink(h2, s0)
        self.addLink(h3, s0)
        self.addLink(h4, s1)

        # Internal Links
        self.addLink(r1, s0)
        self.addLink(s0, s1)
        self.addLink(s1, r2)
        self.addLink(s1, s2)
        self.addLink(s2, r3)

        # WAN Links (r2 <-> r3, redundant)
        self.addLink(r2, r3)
        self.addLink(r2, r3)

def run():
    # Create network
    net = Mininet(topo=SDWANTopo(), controller=RemoteController, link=TCLink, autoSetMacs=True)
    net.start()

    # Configure IPs and routing
    configure_sdwan(net)

    # Enter CLI for testing
    CLI(net)
    net.stop()

if __name__ == '__main__':
    run()
