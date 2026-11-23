from agents.binary_advc import BinaryAdvCAgent
from agents.bc import BCAgent
from agents.csfbc import CSFBCAgent
from agents.dsrl import DSRLAgent
from agents.fbc import FBCAgent
from agents.fsqbc import FSQBCAgent
from agents.fql import FQLAgent
from agents.history_fbc import HistoryFBCAgent
from agents.ifql import IFQLAgent

agents = dict(
    binary_advc=BinaryAdvCAgent,
    bc=BCAgent,
    csfbc=CSFBCAgent,
    dsrl=DSRLAgent,
    fbc=FBCAgent,
    fsqbc=FSQBCAgent,
    fql=FQLAgent,
    history_fbc=HistoryFBCAgent,
    ifql=IFQLAgent,
)
