"""
SQL-схема реляционной базы данных библиотеки бетавольтаических материалов.
"""

CREATE_TABLES_SQL = """
-- 1. Таблица паспортов кристаллических материалов
CREATE TABLE IF NOT EXISTS materials (
    mp_id TEXT PRIMARY KEY,
    formula TEXT NOT NULL,
    crystal_system TEXT,
    density REAL,
    volume REAL,
    e_above_hull REAL,
    formation_energy REAL
);

-- 2. Таблица электрофизических свойств (DFT vs ML калибровка)
CREATE TABLE IF NOT EXISTS electronic_properties (
    mp_id TEXT PRIMARY KEY,
    band_gap_dft REAL,
    delta_eg_predicted REAL,
    band_gap_calibrated REAL,
    eps_ehp_ev REAL,
    Voc_est_v REAL,
    theoretical_efficiency_pct REAL,
    FOREIGN KEY (mp_id) REFERENCES materials (mp_id)
);

-- 3. Таблица радиационной стойкости и бетавольтаических характеристик
CREATE TABLE IF NOT EXISTS betavoltaic_performance (
    mp_id TEXT PRIMARY KEY,
    ed_est_ev REAL,
    radiation_resistance_score REAL,
    
    -- Изотоп Ni-63
    carriers_per_electron_Ni63 REAL,
    penetration_depth_um_Ni63 REAL,
    
    -- Изотоп H-3 (Тритий)
    carriers_per_electron_H3 REAL,
    penetration_depth_um_H3 REAL,
    
    -- Изотоп C-14
    carriers_per_electron_C14 REAL,
    penetration_depth_um_C14 REAL,
    
    -- Изотоп Pm-147 (Прометий)
    carriers_per_electron_Pm147 REAL,
    penetration_depth_um_Pm147 REAL,
    
    FOREIGN KEY (mp_id) REFERENCES materials (mp_id)
);

-- Индексы для мгновенного поиска и фильтрации
CREATE INDEX IF NOT EXISTS idx_formula ON materials(formula);
CREATE INDEX IF NOT EXISTS idx_calibrated_gap ON electronic_properties(band_gap_calibrated);
CREATE INDEX IF NOT EXISTS idx_eff ON electronic_properties(theoretical_efficiency_pct);
CREATE INDEX IF NOT EXISTS idx_rad_score ON betavoltaic_performance(radiation_resistance_score);
"""