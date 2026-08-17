"""
Synthesizes a realistic product catalog.

Why synthesize instead of importing a public dataset: public
Amazon/Flipkart-style scrape dumps carry unclear redistribution licenses and
live at mirror URLs that can disappear — a bad fit for a portfolio repo that
should `git clone` + run reproducibly for anyone, forever. A deterministic
generator (fixed random seed) gives the same ~500 products on every machine
with no external dependency, while still producing varied, realistic-looking
titles/descriptions/prices across categories for hybrid search to work over.

`generate_products()` is a pure function (no Django/DB imports) so it's
unit-testable on its own and reusable from the `seed_catalog` command.
"""

import itertools
import random

# Each category defines the vocabulary combined to build products. Combos
# are drawn from the cartesian product of these lists, so keeping each list
# reasonably sized (5-8 entries) gives thousands of possible combinations
# per category — far more than the ~60 products drawn from it — so products
# within a category read as varied rather than templated.
CATEGORIES = {
    "Footwear": {
        "product_types": [
            "Hiking Boots", "Running Shoes", "Sneakers", "Sandals",
            "Formal Shoes", "Rain Boots", "Trekking Shoes", "Loafers",
        ],
        "brands": [
            "Trailhead", "Norvik", "Stridewell", "Alpine Gear",
            "Urbanstep", "Coastline", "Rockridge",
        ],
        "adjectives": [
            "Waterproof", "Lightweight", "Breathable", "Durable",
            "All-Terrain", "Slip-Resistant", "Cushioned", "Insulated",
        ],
        "materials": ["genuine leather", "synthetic mesh", "rubber sole", "canvas", "GORE-TEX membrane"],
        "colors": ["Black", "Brown", "Grey", "Olive", "Navy", "Tan"],
        "price_range": (999, 6999),
    },
    "Electronics": {
        "product_types": [
            "Wireless Earbuds", "Bluetooth Speaker", "Smartwatch", "Power Bank",
            "Laptop Stand", "USB-C Hub", "Mechanical Keyboard", "Webcam",
        ],
        "brands": [
            "Voltix", "Pulsewave", "Nexbyte", "Circuitry", "Orbital",
            "Quantex", "Ferrotech",
        ],
        "adjectives": [
            "Wireless", "Portable", "Noise-Cancelling", "Fast-Charging",
            "Compact", "Ergonomic", "High-Resolution", "Rugged",
        ],
        "materials": ["aluminum housing", "ABS plastic shell", "braided cable", "silicone grip"],
        "colors": ["Black", "White", "Space Grey", "Silver", "Midnight Blue"],
        "price_range": (499, 8999),
    },
    "Apparel": {
        "product_types": [
            "Denim Jacket", "Cotton T-Shirt", "Hooded Sweatshirt", "Chino Trousers",
            "Wool Sweater", "Track Pants", "Linen Shirt", "Puffer Jacket",
        ],
        "brands": [
            "Urbanloom", "Threadwork", "Cascade Wear", "Meridian",
            "Nomad Co", "Braxton",
        ],
        "adjectives": [
            "Slim-Fit", "Relaxed-Fit", "Breathable", "Stretch", "Classic",
            "Water-Resistant", "Quick-Dry", "Oversized",
        ],
        "materials": ["organic cotton", "merino wool", "recycled polyester", "linen blend", "denim"],
        "colors": ["Charcoal", "Navy", "Olive", "Beige", "Maroon", "White"],
        "price_range": (399, 4499),
    },
    "Home & Kitchen": {
        "product_types": [
            "Non-Stick Frying Pan", "Electric Kettle", "Knife Set", "Air Fryer",
            "Cutting Board", "Storage Container Set", "Coffee Maker", "Blender",
        ],
        "brands": [
            "Heartstone", "Kettlewood", "Culinary Co", "Homestead",
            "Brightware", "Pantryline",
        ],
        "adjectives": [
            "Compact", "Dishwasher-Safe", "Energy-Efficient", "Stainless-Steel",
            "Non-Toxic", "Heavy-Duty", "Space-Saving", "Ergonomic",
        ],
        "materials": ["stainless steel", "borosilicate glass", "cast iron", "BPA-free plastic", "bamboo"],
        "colors": ["Black", "Silver", "White", "Terracotta", "Sage Green"],
        "price_range": (299, 5999),
    },
    "Sports & Outdoors": {
        "product_types": [
            "Yoga Mat", "Camping Tent", "Trekking Backpack", "Water Bottle",
            "Resistance Bands Set", "Sleeping Bag", "Cycling Helmet", "Dumbbell Set",
        ],
        "brands": [
            "Alpine Gear", "Trailhead", "Basecamp", "Vertex Fit",
            "Pinnacle", "Wildpath",
        ],
        "adjectives": [
            "Waterproof", "Lightweight", "Non-Slip", "Foldable", "Insulated",
            "Adjustable", "Heavy-Duty", "Compact",
        ],
        "materials": ["ripstop nylon", "EVA foam", "neoprene", "aluminum alloy", "recycled polyester"],
        "colors": ["Black", "Forest Green", "Red", "Grey", "Blue"],
        "price_range": (499, 7999),
    },
    "Beauty & Personal Care": {
        "product_types": [
            "Facial Cleanser", "Hair Dryer", "Electric Trimmer", "Moisturizer",
            "Sunscreen Lotion", "Hair Straightener", "Electric Toothbrush", "Body Wash",
        ],
        "brands": [
            "Purelume", "Glowbotanics", "Vantique", "Skinlore",
            "Dermavie", "Radiance Lab",
        ],
        "adjectives": [
            "Fragrance-Free", "Dermatologist-Tested", "Fast-Absorbing",
            "Long-Lasting", "Gentle", "Lightweight", "Travel-Size", "Cruelty-Free",
        ],
        "materials": ["aloe vera extract", "hyaluronic acid", "shea butter", "vitamin C complex"],
        "colors": ["N/A"],
        "price_range": (199, 2999),
    },
    "Books": {
        "product_types": [
            "Mystery Novel", "Self-Help Guide", "Science Fiction Novel", "Cookbook",
            "Biography", "Business Strategy Book", "Poetry Collection", "History Book",
        ],
        "brands": [
            "Lighthouse Press", "Cedarbound Books", "Northgate Publishing",
            "Inkwell Editions", "Fernbridge",
        ],
        "adjectives": [
            "Bestselling", "Award-Winning", "Illustrated", "Updated",
            "Unabridged", "Beginner-Friendly", "Critically-Acclaimed", "Compact",
        ],
        "materials": ["paperback", "hardcover", "matte-finish cover"],
        "colors": ["N/A"],
        "price_range": (149, 1499),
    },
    "Toys & Games": {
        "product_types": [
            "Building Block Set", "Board Game", "Remote Control Car", "Puzzle",
            "Plush Toy", "Art Kit", "Educational Tablet", "Card Game",
        ],
        "brands": [
            "Funpeak", "Brightspark", "Tinker Town", "Playnest",
            "Whimsy Works", "Kidcraft",
        ],
        "adjectives": [
            "Educational", "Non-Toxic", "Battery-Operated", "Age-Appropriate",
            "Interactive", "Durable", "Award-Winning", "Compact",
        ],
        "materials": ["ABS plastic", "non-toxic paint", "cotton plush", "recycled cardboard"],
        "colors": ["Multicolor", "Blue", "Pink", "Green", "Red"],
        "price_range": (249, 3499),
    },
}

# Category, adjective and material choice all affect what a description
# reads like, so a single template covers every product without looking
# obviously repeated across categories.
DESCRIPTION_TEMPLATE = (
    "{adjective} {product_type_lower} from {brand}, built with {material}. "
    "{color_clause}Designed for everyday {category_lower} use, combining "
    "practicality with a durable finish that holds up over time."
)


def _price_for(rng: random.Random, price_range: tuple[int, int]) -> str:
    low, high = price_range
    # Round to the nearest 9 (₹999, ₹1499, ...) to read like real retail
    # pricing rather than a uniform random decimal.
    raw = rng.randint(low, high)
    return str(raw - (raw % 10) + 9)


def generate_products(count: int = 500, seed: int = 42) -> list[dict]:
    """Deterministically generate `count` product dicts spread across CATEGORIES.

    Same `seed` always produces the same catalog, which is what makes
    `seed_catalog` idempotent — re-running it regenerates identical
    `external_id`s and upserts rather than duplicating rows.
    """
    rng = random.Random(seed)
    category_names = list(CATEGORIES.keys())
    per_category = count // len(category_names)
    remainder = count - per_category * len(category_names)

    products = []
    for idx, category_name in enumerate(category_names):
        spec = CATEGORIES[category_name]
        n = per_category + (1 if idx < remainder else 0)

        combos = list(itertools.product(
            spec["adjectives"], spec["brands"], spec["product_types"], spec["colors"],
        ))
        rng.shuffle(combos)

        slug = category_name.lower().replace(" & ", "-").replace(" ", "-")
        for i in range(n):
            adjective, brand, product_type, color = combos[i % len(combos)]
            material = rng.choice(spec["materials"])
            title = f"{brand} {adjective} {product_type}" + (f" - {color}" if color != "N/A" else "")
            color_clause = f"Available in {color}. " if color != "N/A" else ""
            description = DESCRIPTION_TEMPLATE.format(
                adjective=adjective,
                product_type_lower=product_type.lower(),
                brand=brand,
                material=material,
                color_clause=color_clause,
                category_lower=category_name.lower(),
            )
            products.append({
                "external_id": f"{slug}-{i + 1:04d}",
                "title": title,
                "description": description,
                "category": category_name,
                "price": _price_for(rng, spec["price_range"]),
                "attributes": {
                    "brand": brand,
                    "material": material,
                    **({"color": color} if color != "N/A" else {}),
                },
            })

    return products
