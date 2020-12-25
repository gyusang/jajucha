# jajucha

[![Build Status]()]()

**jajucha** is a controller Library for jajucha, a model car for autonomous driving education.

- Check Camera Image / LiDAR reading
- process image to infer lanes

# New Features!

- Halt Car inside gui
- Read about/credit pages

## Usage

Check out example.py for examples

### Installation

**jajucha** requires [Python](https://www.python.org/) v3.7+ to run.

Install **jajucha** from [PyPI](https://pypi.org/project/jajucha/) using pip.

```sh
$ pip install jajucha
```

### Todos

- Write Tests

## License

This software is licensed under the MIT License. (
see [LICENSE.txt](https://github.com/gyusang/jajucha/blob/main/LICENSE.txt))

This software includes copy/modification of the following open source software:

- imagezmq, Copyright (c) 2019, Jeff Bass, jeff@yin-yang-ranch.com.
    - in jajucha/communication.py, lines 18-60
    - licensed under
      the [MIT License](https://github.com/jeffbass/imagezmq/blob/7a233584552c929d2a0c7152da563989331735ca/LICENSE.txt)
  > The MIT License (MIT)\
  \
  Copyright (c) 2019, Jeff Bass, jeff@yin-yang-ranch.com.\
  \
  Permission is hereby granted, free of charge, to any person obtaining a copy\
  of this software and associated documentation files (the "Software"), to deal\
  in the Software without restriction, including without limitation the rights\
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\
  copies of the Software, and to permit persons to whom the Software is\
  furnished to do so, subject to the following conditions:\
  \
  The above copyright notice and this permission notice shall be included in\
  all copies or substantial portions of the Software.\
  \
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\
  THE SOFTWARE.

- scipy-cookbook
    - in jajucha/planning.py, lines 180-265
    - licensed under the
      following [license](https://github.com/scipy/scipy-cookbook/blob/5e2833afe5589ac794d70d858a22116144730bee/LICENSE.txt)
  > Copyright (c) 2001, 2002 Enthought, Inc.\
  All rights reserved.\
  \
  Copyright (c) 2003-2017 SciPy Developers.\
  All rights reserved.\
  \
  Redistribution and use in source and binary forms, with or without\
  modification, are permitted provided that the following conditions are met:\
  \
  &nbsp; a. Redistributions of source code must retain the above copyright notice,\
  &nbsp; &nbsp; &nbsp;this list of conditions and the following disclaimer.\
  &nbsp; b. Redistributions in binary form must reproduce the above copyright\
  &nbsp; &nbsp; &nbsp;notice, this list of conditions and the following disclaimer in the\
  &nbsp; &nbsp; &nbsp;documentation and/or other materials provided with the distribution.\
  &nbsp; c. Neither the name of Enthought nor the names of the SciPy Developers\
  &nbsp; &nbsp; &nbsp;may be used to endorse or promote products derived from this software\
  &nbsp; &nbsp; &nbsp;without specific prior written permission.\
  \
  \
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"\
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE\
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE\
  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS\
  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,\
  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF\
  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS\
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN\
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)\
  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF\
  THE POSSIBILITY OF SUCH DAMAGE.
