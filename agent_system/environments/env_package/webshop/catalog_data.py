"""Catalog ownership and the versioned, legacy-compatible seed scheme."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

RENDERER_VERSION = "webshop-native-v3"
SEED_SCHEME = "legacy-isolated-v1"


def native_imports():
    root = str(Path(__file__).parent / "webshop")
    if root not in sys.path:
        sys.path.append(root)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def catalog_fingerprint(config):
    digests = {key: file_digest(config[key]) for key in ("file_path", "attr_path")}
    if config["human_goals"] or config.get("validation"):
        digests["human_attr_path"] = file_digest(config["human_attr_path"])
    key = content_key(dict(digests=digests, human_goals=config["human_goals"], num_products=config["num_products"], synthetic_goal_limit=config.get("synthetic_goal_limit"), renderer_version=RENDERER_VERSION, validation=config.get("validation")))
    return key, digests


@dataclass(frozen=True)
class CatalogMetadata:
    catalog_key: str
    product_count: int
    human_goals: bool
    num_products: int | None
    synthetic_goal_limit: int | None
    max_choices: int
    renderer_version: str = RENDERER_VERSION
    show_attrs: bool = False


@dataclass(frozen=True)
class SeedViewRef:
    catalog_key: str
    seed: int
    price_profile_key: str
    goal_set_key: str
    goal_count: int
    split: str = "train"


@dataclass(frozen=True)
class WebshopSeedView:
    ref: SeedViewRef
    prices: dict
    goals: tuple
    cumulative_weights: tuple


@dataclass(frozen=True)
class WebshopCatalogData:
    """Service-private records. Public service methods return detached subsets only."""

    all_products: list
    product_item_dict: dict
    attribute_to_asins: dict
    metadata: CatalogMetadata
    digests: dict
    validation: dict | None = None

    @classmethod
    def load(cls, config):
        native_imports()
        from web_agent_site.engine.engine import load_catalog_products

        key, digests = catalog_fingerprint(config)
        products, items, attributes = load_catalog_products(config["file_path"], config["attr_path"], config["num_products"], config["human_goals"], config.get("human_attr_path"), include_human_goals=bool(config.get("validation")))
        metadata = CatalogMetadata(key, len(products), config["human_goals"], config["num_products"], config.get("synthetic_goal_limit"), max((len(v) + 1 for p in products for v in p["options"].values()), default=0), show_attrs=config["show_attrs"], renderer_version=RENDERER_VERSION)
        return cls(products, items, attributes, metadata, digests, config.get("validation"))

    def seed_view(self, seed, *, shuffle_goals=True, split="train"):
        from web_agent_site.engine.engine import generate_product_prices
        from web_agent_site.engine.goal import get_goals, get_split_goals

        if split not in {"train", "validation"}:
            raise ValueError(f"Unknown goal split: {split}")
        if split == "validation":
            if not self.validation:
                raise ValueError("Fixed validation is not configured")
            seed = self.validation["seed"]
        rng = random.Random(seed)
        prices = generate_product_prices(self.all_products, rng=rng)
        if self.validation:
            goals = get_split_goals(self.all_products, prices, self.metadata.human_goals, rng, split, self.validation, seed, shuffle_goals, self.metadata.synthetic_goal_limit)
        else:
            goals = get_goals(self.all_products, prices, self.metadata.human_goals, rng=rng, synthetic_goal_limit=self.metadata.synthetic_goal_limit)
        # SimServer explicitly re-seeds here, after generating prices and goals.
        if shuffle_goals and not self.validation:
            rng.seed(seed)
            rng.shuffle(goals)
        key = content_key((self.metadata.catalog_key, seed, SEED_SCHEME))
        goal_order = "goals" if shuffle_goals else "goals-catalog-order-v1"
        if self.validation:
            goal_order = ("fixed-human-split-v1", split, self.validation, shuffle_goals if split == "train" else True)
        ref = SeedViewRef(self.metadata.catalog_key, seed, key, content_key((key, goal_order)), len(goals), split)
        return WebshopSeedView(ref, prices, tuple(goals), tuple(accumulate((g["weight"] for g in goals), initial=0)))
