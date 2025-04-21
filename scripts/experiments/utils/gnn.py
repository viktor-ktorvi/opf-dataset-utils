from omegaconf import DictConfig
from torch import nn
from torch_geometric.nn import GAT, GIN, GraphSAGE


def get_gnn(cfg: DictConfig) -> nn.Module:
    """
    Get a GNN model as specified in the config.

    Parameters
    ----------
    cfg: DictConfig
        Config.

    Returns
    -------
    gnn: nn.Module
    """
    if cfg.training.gnn == "gat":
        return GAT(
            in_channels=cfg.training.hidden_channels,
            edge_dim=-1,
            hidden_channels=cfg.training.hidden_channels,
            num_layers=cfg.training.num_layers,
            out_channels=cfg.training.hidden_channels,
            add_self_loops=False,
            jk=cfg.training.jumping_knowledge,
            v2=True,
            dropout=float(cfg.training.dropout),
            heads=cfg.training.heads,
        )

    if cfg.training.gnn == "gin":
        return GIN(
            in_channels=cfg.training.hidden_channels,
            hidden_channels=cfg.training.hidden_channels,
            num_layers=cfg.training.num_layers,
            out_channels=cfg.training.hidden_channels,
            dropout=float(cfg.training.dropout),
            jk=cfg.training.jumping_knowledge,
        )

    if cfg.training.gnn == "sage":
        return GraphSAGE(
            in_channels=cfg.training.hidden_channels,
            hidden_channels=cfg.training.hidden_channels,
            num_layers=cfg.training.num_layers,
            out_channels=cfg.training.hidden_channels,
            dropout=float(cfg.training.dropout),
            jk=cfg.training.jumping_knowledge,
        )

    raise ValueError(f"GNN '{cfg.training.gnn}' is currently not supported. Expected one of ['gat', 'sage', 'gin']")
