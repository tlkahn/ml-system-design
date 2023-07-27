#!/bin/bash
for file in README*; do
  if [ "$file" != "README.org" ]; then
    rm "$file"
  fi
done