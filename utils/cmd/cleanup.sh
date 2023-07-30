#!/bin/bash
for file in README*; do
  if [ "$file" != "README.org" ] && [ "$file" != "README.trash.org" ]; then
    rm "$file"
  fi
done
