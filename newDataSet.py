#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:04:35 2023

@author: ajusten
"""

import pandas as pd

# Carregue o arquivo CSV em um DataFrame
df = pd.read_csv('Maternal Health Risk Data Set.csv')

# Crie um dicionário de mapeamento para renomear as colunas
novo_nome_colunas = {
    'SystolicBP': 'PressaoSistolica',
    'DiastolicBP': 'PressaoDiastolica',
    'BS': 'NivelDeGlicose',
    'BodyTemp': 'TemperaturaCorporal',
    'HeartRate': 'FrequenciaCardiaca',
    'RiskLevel': 'NivelDeRisco'
}

# Renomeie as colunas usando o método rename
df = df.rename(columns=novo_nome_colunas)

# Salve o DataFrame modificado em um novo arquivo CSV
df.to_csv('Maternal_Health_Risk_Data_Set_Modificado.csv', index=False)
