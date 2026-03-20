# Product and Store Catalog for InventoryIQ

# 50 Products (Retail/Electronics/Sports mix)
PRODUCTS = {
    1: "Laptop Dell XPS 13",
    2: "Mouse Logitech MX Master",
    3: "Teclado Mecánico Corsair",
    4: "Monitor Samsung 27\"",
    5: "Auriculares Sony WH-1000XM5",
    6: "Webcam Logitech C920",
    7: "SSD Samsung 1TB",
    8: "RAM Corsair 16GB DDR4",
    9: "Tarjeta Gráfica RTX 3060",
    10: "Procesador Intel i7",
    11: "Zapatillas Running Nike",
    12: "Camiseta Deportiva Adidas",
    13: "Pantalón Jogger Puma",
    14: "Mochila North Face",
    15: "Botella Térmica Hydro Flask",
    16: "Yoga Mat Premium",
    17: "Pesas Ajustables 20kg",
    18: "Banda Elástica Resistencia",
    19: "Reloj Deportivo Garmin",
    20: "Gafas de Sol Oakley",
    21: "Cafetera Nespresso",
    22: "Licuadora Ninja",
    23: "Freidora de Aire Philips",
    24: "Microondas Samsung",
    25: "Tostadora Cuisinart",
    26: "Batidora KitchenAid",
    27: "Olla Arrocera Zojirushi",
    28: "Juego de Cuchillos Wüsthof",
    29: "Sartén Antiadherente Tefal",
    30: "Termo Stanley 1L",
    31: "Smartphone Samsung Galaxy",
    32: "iPhone 15 Pro",
    33: "Tablet iPad Air",
    34: "Smartwatch Apple Watch",
    35: "Cargador Inalámbrico Anker",
    36: "Power Bank 20000mAh",
    37: "Cable USB-C Belkin",
    38: "Funda Protectora Spigen",
    39: "Protector de Pantalla",
    40: "Soporte para Celular",
    41: "Libro \"Hábitos Atómicos\"",
    42: "Cuaderno Moleskine",
    43: "Bolígrafos Pilot G2",
    44: "Marcadores Sharpie",
    45: "Agenda 2026 Leuchtturm",
    46: "Lámpara de Escritorio LED",
    47: "Silla Ergonómica Herman Miller",
    48: "Escritorio Ajustable",
    49: "Organizador de Cables",
    50: "Planta Suculenta Decorativa"
}

# 10 Stores (Geographic distribution)
STORES = {
    1: "Tienda Centro - Santiago",
    2: "Sucursal Mall Plaza - Viña",
    3: "Local Costanera Center - Stgo",
    4: "Tienda Parque Arauco - Las Condes",
    5: "Sucursal Portal La Dehesa",
    6: "Local Mall Sport - Ñuñoa",
    7: "Tienda Alto Las Condes",
    8: "Sucursal La Florida",
    9: "Local Maipú Centro",
    10: "Tienda Outlet - Quilicura"
}

if __name__ == "__main__":
    print("=== CATÁLOGO DE PRODUCTOS ===")
    for id, name in PRODUCTS.items():
        print(f"{id:2d}. {name}")
    
    print("\n=== CATÁLOGO DE TIENDAS ===")
    for id, name in STORES.items():
        print(f"{id:2d}. {name}")
