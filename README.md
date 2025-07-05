# TraRyTrade_SelfSwimmBot_V1

TraRyTrade_SelfSwimm_V1__PublicOpenSourceRelease_SuperRefined_MultiLot_Async - Single Unified Script with Extended ML Actions  


set the config.py you API Stuff 
start with python3 TraRyTrade_SelfSwimm_V1.py  (may there lot of missing libs )


## 1. Quick Setup

1. **Clone or copy** this repo into a folder.
 
2. **Create a Python virtual environment** (recommended):
python3 -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install \
  aiohttp \
  numpy \
  pandas \
  joblib \
  scikit-learn \
  websockets \
  python-binance

May some more libs needed, base on your system. 
Go through the message you receive when you try to start the bot. 
You may need to search on Google for any missing libraries and install them using pip install until you can successfully run the bot. 

I apologize; I will add a more precise list later. However, for normal Python users, figuring out the missing libraries shouldn't be a problem.


3. Configure your API Keys
Set in config.py your binance keys. 

# config.py
api_key_binance    = "YOUR_BINANCE_API_KEY"
api_secret_binance = "YOUR_BINANCE_API_SECRET"

Set in varmove.py Symbol and TradeAmount and * TradeX 
# varmove.py
Set here Coin You wana trade and TradeX  // exp 400 x 24  for both site sellrun or buyrun....

exp: varmove.py
Coin = "LUMIAUSDT"
TradeAmount = 24
TradeX = 50      # 400+ may 2000+ or more is better at last but much risk on binance very expensive 


4. Run the Bot
python3 TraRyTrade_SelfSwimm_V1.py


⚠️ ATTENTION

You may need to tweak setVars (e.g. symbol, lot sizes, thresholds) for your market and risk profile.
Out of the box it may not be profitable—use at your own risk! And in beagnn when model fresh empty it is much bader as later....



