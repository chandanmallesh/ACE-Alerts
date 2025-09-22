import importlib.util, sys, pathlib
    
spec = importlib.util.spec_from_file_location("ace_main", str(pathlib.Path(__file__).parent / "main.py"))
ace = importlib.util.module_from_spec(spec)
sys.modules["ace_main"] = ace
spec.loader.exec_module(ace)

SAMPLE = "Premarket Top % Gainers: $AGMH +230%, $CJET +150%, $ATMV +68%, $ADAP +19%, $YDKG +4%"

def test_parser_extracts_pairs():
    pairs = ace.parse_tickers_with_pct(SAMPLE)
    assert ("AGMH", 230.0) in pairs and ("CJET", 150.0) in pairs

def test_rank_has_premarket():
    pairs = ace.parse_tickers_with_pct(SAMPLE)
    df = ace.rank_tickers(pairs)
    assert "Premarket%" in df.columns and df["Premarket%"].sum() >= 471.0
