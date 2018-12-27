#!/usr/bin/env bash
if [ "$TRAVIS_BRANCH" = 'master' ] && [ "$TRAVIS_PULL_REQUEST" == 'false' ]; then
    openssl aes-256-cbc -K $encrypted_9c3484b2e90f_key -iv $encrypted_9c3484b2e90f_iv -in cd/codesigning.asc.enc -out cd/codesigning.asc -d
    gpg --fast-import cd/codesigning.asc
    mvn deploy -P sign,build-extras,lolopy --settings cd/mvnsettings.xml
fi
