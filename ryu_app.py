# ryu_app.py (Updated for Q-learning model)
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4
import predict
import time

class SDWANRLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SDWANRLController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.packet_count = 0
        
        # Load Q-learning model
        self.model_loaded = predict.load_model()
        if self.model_loaded:
            self.logger.info(f"[INFO] Q-learning model loaded successfully: {predict.get_model_info()}")
        else:
            self.logger.warning("[WARN] Running with random action fallback")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        self.packet_count += 1
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        self.logger.info("PacketIn #%d: %s → %s (DPID: %s, Port: %s)", 
                        self.packet_count, src, dst, dpid, in_port)

        try:
            # Get network state
            state = predict.get_network_state(datapath, in_port)
            
            # Get Q-learning action
            action = predict.select_action(state)
            
            self.logger.info(f"Q-Learning Decision - State: {state.tolist()[:2]} | Action: {action} | Model: {self.model_loaded}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Q-learning prediction failed: {e}")
            action = 0  # fallback

        # Apply Q-learning decision to routing
        if dst in self.mac_to_port[dpid]:
            if action == 1:
                # Action 1: Direct forwarding (exploitation)
                out_port = self.mac_to_port[dpid][dst]
                self.logger.info(f"Q-Action 1: Direct forwarding to port {out_port}")
            else:
                # Action 0: Broadcast for exploration/discovery
                out_port = ofproto.OFPP_FLOOD
                self.logger.info("Q-Action 0: Broadcasting for network exploration")
        else:
            # Unknown destination - always flood
            out_port = ofproto.OFPP_FLOOD
            self.logger.info("Unknown destination: Broadcasting")

        actions = [parser.OFPActionOutput(out_port)]

        # Install flow rule for direct forwarding only
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)

        # Send packet out
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=in_port, actions=actions, data=msg.data)
        datapath.send_msg(out)
        
        # Adaptive epsilon (optional - gradually reduce exploration over time)
        if self.packet_count % 100 == 0:  # Every 100 packets
            new_epsilon = max(0.1, 0.9 * (0.99 ** (self.packet_count // 100)))
            predict.update_epsilon(new_epsilon)
