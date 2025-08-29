from src.models.collab_matrix import SimpleMF
import pandas as pd

def test_collab_basic():
    df = pd.DataFrame({
        "user_id": ["u1","u1","u2","u2"],
        "item_id": ["i1","i2","i1","i3"],
        "count": [1,1,1,1]
    })
    mf = SimpleMF(n_factors=2)
    mf.fit(df)
    recs = mf.recommend("u1", top_k=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2
