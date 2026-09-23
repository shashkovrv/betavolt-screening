import sqlite3
import pandas as pd

conn = sqlite3.connect('data/05_database/betavoltaic_library.db')

print("--- mp-66 and mp-149 ---")
sql1 = """
SELECT m.mp_id, m.formula, m.crystal_system, m.density, m.e_above_hull, 
       e.band_gap_dft, e.band_gap_calibrated, p.radiation_resistance_score 
FROM materials m 
JOIN electronic_properties e ON m.mp_id=e.mp_id 
JOIN betavoltaic_performance p ON m.mp_id=p.mp_id 
WHERE m.mp_id IN ('mp-66', 'mp-149')
"""
print(pd.read_sql_query(sql1, conn))

print("\n--- All materials with formula C ---")
sql2 = """
SELECT m.mp_id, m.formula, m.crystal_system, m.density, m.e_above_hull, 
       e.band_gap_dft, e.band_gap_calibrated, p.radiation_resistance_score 
FROM materials m 
JOIN electronic_properties e ON m.mp_id=e.mp_id 
JOIN betavoltaic_performance p ON m.mp_id=p.mp_id 
WHERE m.formula = 'C'
"""
print(pd.read_sql_query(sql2, conn))

print("\n--- All materials with formula Si ---")
sql3 = """
SELECT m.mp_id, m.formula, m.crystal_system, m.density, m.e_above_hull, 
       e.band_gap_dft, e.band_gap_calibrated, p.radiation_resistance_score 
FROM materials m 
JOIN electronic_properties e ON m.mp_id=e.mp_id 
JOIN betavoltaic_performance p ON m.mp_id=p.mp_id 
WHERE m.formula = 'Si'
"""
print(pd.read_sql_query(sql3, conn))
