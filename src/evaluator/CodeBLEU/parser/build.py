# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'C:\Users\ntlinh\OneDrive - Hanoi University of Science and Technology\Documents\20242\PP\src\evaluator\CodeBLEU\parser\my-languages.so',

  # Include one or more languages
  [
    'tree-sitter-go',
    'tree-sitter-javascript',
    'tree-sitter-python',
    #'tree-sitter-php',
    'tree-sitter-java',
    'tree-sitter-ruby',
    'tree-sitter-c-sharp',
  ]
)

