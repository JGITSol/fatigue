Versioning Strategy
===================

This project follows `Semantic Versioning 2.0.0 <https://semver.org/>`_ with additional pre-release tags to indicate development stages.

Version Format
-------------

::

    MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

* **MAJOR**: Incremented for incompatible API changes
* **MINOR**: Incremented for backward-compatible functionality additions
* **PATCH**: Incremented for backward-compatible bug fixes
* **PRERELEASE**: Optional tag indicating pre-release status (e.g., alpha, beta, rc)
* **BUILD**: Optional build metadata

Current Version
--------------

The current version is: **|release|**

Development Stages
-----------------

Alpha Stage (0.1.0-alpha)
~~~~~~~~~~~~~~~~~~~~~~~~~
* Initial development
* Core functionality implementation
* Not feature complete
* Expect bugs and API changes

Beta Stage (0.x.0-beta)
~~~~~~~~~~~~~~~~~~~~~~
* Feature complete for the planned release
* Testing and stabilization
* API may still change

Release Candidate (0.x.0-rc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Feature and API frozen
* Bug fixing only
* Preparing for final release

Release (1.0.0+)
~~~~~~~~~~~~~~
* Stable release
* Production ready

Version Control with Git
-----------------------

We use Git tags to mark version releases::

    # For the current alpha version
    git tag -a v0.1.0-alpha -m "Initial alpha release"
    
    # Push tags to remote repository
    git push origin --tags

Version Bumping Guidelines
-------------------------

* **PATCH**: Bug fixes that don't change the API
* **MINOR**: New features that are backward compatible
* **MAJOR**: Changes that break backward compatibility

Documentation and Testing
------------------------

Each version should include:

* Updated documentation with Sphinx
* Test coverage reports
* Changelog entries