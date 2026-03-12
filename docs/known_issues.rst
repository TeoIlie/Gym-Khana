.. image:: assets/logo.png
   :width: 20
   :align: left
   :alt: logo

Known Issues
============

Windows
-------

.. warning::

   Library support issues on Windows. You must use Python 3.8 as of 10-2021.

macOS Big Sur and above
-----------------------

.. warning::

   When rendering is turned on, you might encounter:

   .. code::

      ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.

   Fix by installing a newer version of pyglet:

   .. code:: bash

      pip3 install pyglet==1.5.11

   You might see a warning like:

   .. code::

      gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.

   This can be safely ignored. The environment will still work without error.
