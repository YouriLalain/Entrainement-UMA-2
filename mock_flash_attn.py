#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock pour flash_attn qui n'est pas compatible avec Mac
Ce module permet de contourner l'erreur d'import d'OVIS
"""

import sys
from unittest.mock import MagicMock

# Crée un mock module pour flash_attn
flash_attn_mock = MagicMock()
flash_attn_mock.flash_attn_func = None
flash_attn_mock.flash_attn_varlen_func = None

sys.modules['flash_attn'] = flash_attn_mock
sys.modules['flash_attn.flash_attn_interface'] = flash_attn_mock

print("✅ Mock flash_attn créé pour compatibilité Mac")
