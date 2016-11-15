
def hybrid_rec(imf_pred, ub_pred, alpha=0.8):
    ### Combined predictions
    hyb_pred = alpha * imf_pred + (1 - alpha) * ub_pred

    return hyb_pred
