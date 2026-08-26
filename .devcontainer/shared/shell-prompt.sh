# shellcheck shell=bash
# Git-aware prompt and tab completion for the mononet devcontainer.
#
# Sourced from ~/.bashrc and ~/.zshrc by install-shell-prompt.sh. Replaces the
# stock "user@host:cwd$" prompt with "cwd (branch)$" -- the user and host are
# noise inside a container, the branch is not -- and wires up completion for
# the CLIs that do not ship a completion file (gh, uv), fzf key bindings,
# EDITOR/PAGER, and history defaults. The packages behind this are installed by
# shared/install_common_tools.sh.
#
# A virtualenv prefix (e.g. "(mononet) ") is still prepended by the activate
# script via VIRTUAL_ENV_PROMPT / PS1 rewriting, so it survives this.

# The image has no editor and no pager alternative until
# install_common_tools.sh runs; git then resolves core.editor -> EDITOR ->
# /usr/bin/editor. Set these only when the caller has not.
if [ -z "${EDITOR:-}" ] && command -v vim >/dev/null 2>&1; then
    EDITOR=vim
    export EDITOR
fi
[ -z "${VISUAL:-}" ] && [ -n "${EDITOR:-}" ] && export VISUAL="$EDITOR"
# -F: skip the pager for output that fits one screen. -R: keep git/gh colors.
# -X: leave the output on screen after quitting. Matches git's own default.
[ -z "${LESS:-}" ] && export LESS="-FRX"

# The stock rc keeps 1000 lines with no timestamps, which is not much history
# for a long-lived container.
HISTSIZE=100000
HISTFILESIZE=200000
HISTCONTROL=ignoreboth:erasedups
HISTTIMEFORMAT='%F %T '
export HISTSIZE HISTFILESIZE HISTCONTROL HISTTIMEFORMAT

# Cache dir for generated completion scripts (gh/uv emit them on stdout;
# regenerating on every shell start would cost a subprocess each).
_mononet_comp_cache="${XDG_CACHE_HOME:-$HOME/.cache}/mononet-shell"

# _mononet_completion <tool> <shell> <arg>...
# Cache `<tool> <arg>...` output once, then source it.
_mononet_completion() {
    local tool=$1 shell=$2
    shift 2
    command -v "$tool" >/dev/null 2>&1 || return 0
    local cache="$_mononet_comp_cache/$tool.$shell"
    if [ ! -s "$cache" ]; then
        mkdir -p "$_mononet_comp_cache"
        "$tool" "$@" > "$cache" 2>/dev/null || { rm -f "$cache"; return 0; }
    fi
    # shellcheck disable=SC1090
    . "$cache"
}

if [ -n "${BASH_VERSION:-}" ]; then

    # The stock ~/.bashrc sources the bash-completion loader when present, but
    # only some images ship it; load it here too so this file works standalone.
    # (install_common_tools.sh apt-installs the package.)
    if ! declare -F _completion_loader >/dev/null 2>&1 && ! shopt -oq posix; then
        for _mononet_bc in \
            /usr/share/bash-completion/bash_completion \
            /etc/bash_completion
        do
            # shellcheck disable=SC1090
            [ -r "$_mononet_bc" ] && . "$_mononet_bc" && break
        done
        unset _mononet_bc
    fi

    _mononet_completion gh bash completion -s bash
    _mononet_completion uv bash generate-shell-completion bash

    # Debian ships __git_ps1 in git-sh-prompt, but the container has no
    # bash-completion loader to source it, so do it ourselves.
    if ! declare -F __git_ps1 >/dev/null 2>&1; then
        for _mononet_gitprompt in \
            /usr/lib/git-core/git-sh-prompt \
            /usr/share/git-core/contrib/completion/git-prompt.sh \
            /etc/bash_completion.d/git-prompt
        do
            # shellcheck disable=SC1090
            [ -r "$_mononet_gitprompt" ] && . "$_mononet_gitprompt" && break
        done
        unset _mononet_gitprompt
    fi

    if ! declare -F __git_ps1 >/dev/null 2>&1; then
        # Fallback: no git-sh-prompt available in this image.
        __git_ps1() {
            local branch
            branch=$(git symbolic-ref --short -q HEAD 2>/dev/null) ||
                branch=$(git rev-parse --short HEAD 2>/dev/null) || return 0
            printf "${1:- (%s)}" "$branch"
        }
    fi

    GIT_PS1_SHOWDIRTYSTATE=1      # * unstaged, + staged
    GIT_PS1_SHOWSTASHSTATE=1      # $ stashed
    GIT_PS1_SHOWUNTRACKEDFILES=1  # % untracked
    GIT_PS1_SHOWUPSTREAM=auto     # < behind, > ahead, = in sync
    export GIT_PS1_SHOWDIRTYSTATE GIT_PS1_SHOWSTASHSTATE \
        GIT_PS1_SHOWUNTRACKEDFILES GIT_PS1_SHOWUPSTREAM

    # __git_ps1 appends the upstream marker last, and "=" (in sync) is the
    # normal state -- worth no pixels. Drop it and keep < / > / <> , which do
    # say something. Everything else passes through untouched.
    __mononet_git_ps1() {
        local s
        s=$(__git_ps1 "%s") || return 0
        [ -n "$s" ] || return 0
        printf ' (%s)' "${s%=}"
    }

    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;34m\]\w\[\033[00m\]\[\033[01;33m\]$(__mononet_git_ps1)\[\033[00m\]\$ '

    case "$TERM" in
        xterm* | rxvt* | screen* | tmux*)
            PS1="\[\e]0;\w\a\]$PS1"
            ;;
    esac

    shopt -s histappend globstar

    # Readline: Up/Down search history by what is already typed rather than
    # walking it blindly; completion ignores case and lists on first ambiguity.
    if [[ $- == *i* ]]; then
        bind '"\e[A": history-search-backward' 2>/dev/null
        bind '"\e[B": history-search-forward' 2>/dev/null
        bind 'set completion-ignore-case on' 2>/dev/null
        bind 'set show-all-if-ambiguous on' 2>/dev/null
        bind 'set colored-stats on' 2>/dev/null
    fi

    # fzf: Ctrl-R fuzzy history, Ctrl-T file picker, Alt-C cd. The Debian
    # package ships the bindings as an example file rather than sourcing them.
    # Must come after the bind calls above -- fzf rebinds Ctrl-R.
    # shellcheck disable=SC1091
    [ -r /usr/share/doc/fzf/examples/key-bindings.bash ] &&
        . /usr/share/doc/fzf/examples/key-bindings.bash

elif [ -n "${ZSH_VERSION:-}" ]; then

    # oh-my-zsh runs compinit itself; only do it when nothing else has.
    if ! whence -w compdef >/dev/null 2>&1; then
        autoload -Uz compinit && compinit -u
    fi
    zstyle ':completion:*' menu select
    zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'

    _mononet_completion gh zsh completion -s zsh
    _mononet_completion uv zsh generate-shell-completion zsh

    setopt APPEND_HISTORY SHARE_HISTORY HIST_IGNORE_ALL_DUPS \
        HIST_IGNORE_SPACE EXTENDED_GLOB
    HISTFILE="${HISTFILE:-$HOME/.zsh_history}"
    SAVEHIST=$HISTFILESIZE

    # Same prefix-search-on-arrow behaviour as the bash half.
    autoload -Uz up-line-or-beginning-search down-line-or-beginning-search
    zle -N up-line-or-beginning-search
    zle -N down-line-or-beginning-search
    bindkey '^[[A' up-line-or-beginning-search
    bindkey '^[[B' down-line-or-beginning-search

    for _mononet_fzf in /usr/share/doc/fzf/examples/completion.zsh \
        /usr/share/doc/fzf/examples/key-bindings.zsh
    do
        # shellcheck disable=SC1090
        [ -r "$_mononet_fzf" ] && . "$_mononet_fzf"
    done
    unset _mononet_fzf

    autoload -Uz vcs_info
    zstyle ':vcs_info:*' enable git
    zstyle ':vcs_info:git:*' check-for-changes true
    zstyle ':vcs_info:git:*' unstagedstr '*'
    zstyle ':vcs_info:git:*' stagedstr '+'
    zstyle ':vcs_info:git:*' formats       ' (%b%u%c)'
    zstyle ':vcs_info:git:*' actionformats ' (%b|%a%u%c)'

    _mononet_precmd() { vcs_info; }
    autoload -Uz add-zsh-hook
    add-zsh-hook precmd _mononet_precmd

    setopt PROMPT_SUBST
    PROMPT='%F{blue}%~%f%F{yellow}${vcs_info_msg_0_}%f%# '

fi
